"""Stage 7: controlled EHR-anchored, frozen-feature TRCG refinement.

- No test data are loaded.
- Stage 2 encoders stay frozen in cached representations.
- The pretrained EHR mortality head is frozen, and the last CXR-residual
  linear layer is zero-initialized. Initial fused predictions must match
  EHR-only predictions exactly (checked in smoke).
- Compare a learned gate with an equally scaled constant gate from TRAIN,
  and weak proposal-defined age x EHR-change regularization.

python scripts/18_train_stage7_refinement.py --variant safe_learned --smoke
python scripts/18_train_stage7_refinement.py --variant safe_learned --seed 42 --resume
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.trcg_fusion import TRCGFusion

CACHE = ROOT / 'cache'
CKPT = ROOT / 'checkpoints'
RESULTS = ROOT / 'results'
LOGS = ROOT / 'logs'
for d in (CKPT, RESULTS, LOGS):
    d.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VARIANTS = ('safe_learned', 'safe_fixed', 'safe_age')
FEATURES = ('ehr_at_48', 'ehr_at_cxr', 'cxr', 'cxr_age',
            'anchor_valid', 'ehr_quality', 'cxr_entropy')


class Cached(Dataset):
    def __init__(self, split):
        path = CACHE / f'trcg_stage4_{split}_features.npz'
        if not path.exists():
            raise FileNotFoundError(f'Missing cached data {path} (run Stage 4A prepare)')
        with np.load(path) as arr:
            self.inputs = {k: torch.tensor(arr[k], dtype=torch.float32) for k in FEATURES}
            self.y = torch.tensor(arr['label'], dtype=torch.float32)
            self.ids = [str(x) for x in arr['stay_id']]
        assert len(self.y) == len(self.ids) and len(self.y) > 0
        assert len(set(self.ids)) == len(self.ids), f'duplicate {split} stay_id'
        assert all(torch.isfinite(x).all() for x in self.inputs.values())
        assert ((self.y == 0) | (self.y == 1)).all()
        assert self.inputs['ehr_at_48'].shape[1] == 128
        assert self.inputs['ehr_at_cxr'].shape[1] == 128
        assert self.inputs['cxr'].shape[1] == 512
        assert self.inputs['ehr_quality'].shape[1] == 5

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return ({k: v[i] for k, v in self.inputs.items()}, self.y[i], self.ids[i])


def atomic_save(blob, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(blob, tmp)
    os.replace(tmp, path)


def atomic_json(blob, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(blob, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_states():
    es = torch.load(CKPT / 'trcg_stage2_ehr_seed42_best.pt', map_location='cpu', weights_only=True)['model']
    cs = torch.load(CKPT / 'trcg_stage2_cxr_seed42_best.pt', map_location='cpu', weights_only=True)['model']
    al = torch.load(CKPT / 'trcg_stage3_align_seed42_best.pt', map_location='cpu', weights_only=True)['model']
    esh = {'weight': es['head.weight'], 'bias': es['head.bias']}
    csh = {'weight': cs['head.weight'], 'bias': cs['head.bias']}
    return esh, csh, al


class SafeTRCG(TRCGFusion):
    def __init__(self, esh, csh, alignment, variant, constant_gamma):
        super().__init__(esh, csh, alignment)
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.constant_gamma = float(constant_gamma)
        # Preserve EHR-only prediction at initialization; the correction
        # learns gradually rather than disrupting the pretrained EHR head.
        nn.init.zeros_(self.residual[-2].weight)
        nn.init.zeros_(self.residual[-2].bias)
        self.ehr_head.requires_grad_(False)
        self.cxr_head.requires_grad_(False)
        self.change.requires_grad_(False)  # no-change experiments
        if variant == 'safe_fixed':
            self.gate.requires_grad_(False)

    def forward(self, b):
        zt, za, zc = b['ehr_at_48'], b['ehr_at_cxr'], b['cxr']
        age = (b['cxr_age'].view(-1, 1) / 48.).clamp(0, 1)
        valid = b['anchor_valid'].view(-1, 1).float()
        entropy = b['cxr_entropy'].view(-1, 1)
        qe, qc = self.projections(za, zc)
        agreement = torch.cat([qc, qe, (qc-qe).abs(), qc*qe], dim=-1)
        d = torch.zeros_like(zt)  # remove learned post-CXR change component
        re = torch.sigmoid(self.ehr_reliability(b['ehr_quality']))
        rc = torch.sigmoid(self.cxr_reliability(torch.cat(
            [age, entropy, self.image_quality(zc)], dim=-1)))
        delta = self.residual(torch.cat([zc, za, d, agreement], dim=-1))
        if self.variant == 'safe_fixed':
            gamma = torch.full_like(age, self.constant_gamma)
        else:
            gamma = torch.sigmoid(self.gate(torch.cat(
                [agreement, d, age, re, rc, valid], dim=-1)))
        # Missing pre-CXR EHR anchor: avoid invalid cross-modal concordance.
        effective = gamma * rc * valid
        fused = zt + effective * delta
        fused_logit = self.ehr_head(fused).squeeze(-1)
        ehr_logit = self.ehr_head(zt).squeeze(-1)
        return dict(logits=fused_logit, ehr_logits=ehr_logit,
                    gamma=gamma.squeeze(-1), r_cxr=rc.squeeze(-1),
                    effective=effective.squeeze(-1), delta=delta)


def compute_train_gamma(esh, csh, align):
    """Measure matched constant gate from TRAIN using original Stage-4A model."""
    path = CKPT / 'trcg_stage4a_v2_seed42_best.pt'
    if not path.exists():
        raise FileNotFoundError(f'Missing original Stage4A checkpoint {path}')
    original = TRCGFusion(esh, csh, align).to(DEVICE).eval()
    original.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'], strict=True)
    loader = DataLoader(Cached('train'), batch_size=256, shuffle=False)
    val, n = 0., 0
    with torch.no_grad():
        for x, y, ids in loader:
            x = {k: v.to(DEVICE) for k, v in x.items()}
            gamma = original(x)['gamma']
            val += gamma.sum().item()
            n += len(gamma)
    return val / n


def objective(model_output, x, y, pos_weight, variant):
    mort = nn.functional.binary_cross_entropy_with_logits(
        model_output['logits'], y, pos_weight=pos_weight)
    penalty = mort.new_zeros(())
    if variant == 'safe_age':
        with torch.no_grad():
            clinical_change = (x['ehr_at_48'] - x['ehr_at_cxr']).norm(dim=-1)
            clinical_change = clinical_change / (clinical_change + 3.)
            age = (x['cxr_age'] / 48.).clamp(0, 1)
            prior = age * clinical_change * x['anchor_valid']
        penalty = (model_output['gamma'] * prior).mean()
    return mort + (0.02 * penalty if variant == 'safe_age' else 0.), mort, penalty


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_all, p_all, base, ages, weights, magnitudes = [], [], [], [], [], []
    for x, y, ids in loader:
        x = {k: v.to(DEVICE) for k, v in x.items()}
        out = model(x)
        p_all.extend(torch.sigmoid(out['logits']).cpu().tolist())
        base.extend(torch.sigmoid(out['ehr_logits']).cpu().tolist())
        y_all.extend(y.tolist())
        ages.extend(x['cxr_age'].detach().cpu().flatten().tolist())
        weights.extend(out['effective'].cpu().tolist())
        magnitudes.extend((out['logits'] - out['ehr_logits']).abs().cpu().tolist())
    ap = float(average_precision_score(y_all, p_all))
    auc = float(roc_auc_score(y_all, p_all))
    base_ap = float(average_precision_score(y_all, base))
    base_auc = float(roc_auc_score(y_all, base))
    corr = None
    if np.std(ages) > 0 and np.std(weights) > 0:
        from scipy.stats import spearmanr
        v = float(spearmanr(ages, weights).statistic)
        if np.isfinite(v):
            corr = v
    return dict(val_auprc=ap, val_auroc=auc,
                ehr_base_auprc=base_ap, ehr_base_auroc=base_auc,
                mean_abs_logit_shift=float(np.mean(magnitudes)),
                mean_effective_cxr_weight=float(np.mean(weights)),
                spearman_effective_vs_cxr_age=corr)


def save_history(path, rows):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss',
            'val_auprc', 'val_auroc', 'ehr_base_auprc', 'mean_abs_logit_shift'])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def run(args):
    seed_all(args.seed)
    tr, va = Cached('train'), Cached('val')
    if set(tr.ids) & set(va.ids):
        raise RuntimeError('Train/validation stay IDs overlap')
    train = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    val = DataLoader(va, batch_size=256, shuffle=False)
    esh, csh, align = source_states()
    fixed_gamma = compute_train_gamma(esh, csh, align)
    model = SafeTRCG(esh, csh, align, args.variant, fixed_gamma).to(DEVICE)
    positive = float(tr.y.sum())
    weight = torch.tensor((len(tr)-positive)/positive, device=DEVICE)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=1e-4)
    prefix = f'trcg_stage7_{args.variant}_seed{args.seed}'
    last = CKPT / f'{prefix}_last.pt'
    best = CKPT / f'{prefix}_best.pt'
    done = CKPT / f'{prefix}.done'
    summary_path = RESULTS / f'{prefix}_summary.json'
    history_path = LOGS / f'{prefix}_history.csv'
    settings = dict(variant=args.variant, seed=args.seed,
                    lr=args.lr, batch_size=args.batch_size,
                    epochs=args.epochs, patience=args.patience,
                    gamma_from_stage4a_train=round(fixed_gamma, 8),
                    frozen_ehr_head=True, zero_init_residual=True,
                    no_change=True, no_align_loss=True)
    # Critical invariant before updates: fusion = pretrained EHR.
    model.eval()
    x0, y0, _ = next(iter(val))
    with torch.no_grad():
        out0 = model({k: v.to(DEVICE) for k, v in x0.items()})
    difference = float((out0['logits']-out0['ehr_logits']).abs().max())
    if difference > 1e-5:
        raise RuntimeError(f'Initial EHR anchor NOT preserved: {difference}')
    print(f'EHR anchor check PASSED; max initial logit diff={difference:.2e}; '
          f'train-derived fixed gamma={fixed_gamma:.5f}', flush=True)
    if args.smoke:
        model.train()
        x,y,_ = next(iter(train))
        x = {k:v.to(DEVICE) for k,v in x.items()}
        out = model(x)
        loss,mort,penalty = objective(out, x, y.to(DEVICE), weight, args.variant)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite smoke loss')
        print(f'SMOKE PASSED {args.variant}: loss={loss.item():.4f}', flush=True)
        return
    if done.exists():
        old = json.loads(done.read_text(encoding='utf-8'))
        if old['settings'] != settings:
            raise RuntimeError(f'{done} exists with other settings')
        print('ALREADY COMPLETED', prefix, flush=True)
        return
    start,best_ap,best_auc,best_ep,bad,hist=1,-1.,-1.,0,0,[]
    if args.resume and last.exists():
        ck = torch.load(last, map_location=DEVICE, weights_only=False) # local checkpoint
        if ck['settings'] != settings:
            raise RuntimeError(f'Incompatible resume file: {last}')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        start,best_ap,best_auc,best_ep,bad,hist = [ck[k] for k in
            ('epoch_next','best_ap','best_auc','best_epoch','bad','history')]
        random.setstate(ck['python_rng'])
        np.random.set_state(ck['numpy_rng'])
        torch.random.set_rng_state(ck['torch_rng'])
        if torch.cuda.is_available() and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        print('RESUME', prefix, 'epoch', start, flush=True)
    for epoch in range(start, args.epochs+1):
        if bad >= args.patience:
            print('Early stop already reached', flush=True)
            break
        model.train()
        losses=[]
        for x,y,_ in train:
            x={k:v.to(DEVICE) for k,v in x.items()}
            y=y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss, mort, reg = objective(out,x,y,weight,args.variant)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],1.)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        metrics=evaluate(model,val)
        is_better = metrics['val_auprc'] > best_ap + 1e-6 or (
            abs(metrics['val_auprc']-best_ap) <=1e-6 and
            metrics['val_auroc']>best_auc+1e-6)
        if is_better:
            best_ap=metrics['val_auprc'];best_auc=metrics['val_auroc'];best_ep=epoch;bad=0
            atomic_save({'model':model.state_dict(),'epoch':epoch,'metrics':metrics,'settings':settings},best)
        else:
            bad+=1
        hist.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),
            val_auprc=metrics['val_auprc'],val_auroc=metrics['val_auroc'],
            ehr_base_auprc=metrics['ehr_base_auprc'],
            mean_abs_logit_shift=metrics['mean_abs_logit_shift']))
        atomic_save(dict(model=model.state_dict(),optimizer=opt.state_dict(),
            epoch_next=epoch+1,best_ap=best_ap,best_auc=best_auc,best_epoch=best_ep,bad=bad,
            history=hist,settings=settings,python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),torch_rng=torch.random.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),last)
        save_history(history_path,hist)
        print(f'{prefix} epoch {epoch}: val AUROC={metrics["val_auroc"]:.4f} '
              f'AUPRC={metrics["val_auprc"]:.4f} | EHR anchor AUPRC={metrics["ehr_base_auprc"]:.4f} '
              f'best={best_ep} bad={bad}/{args.patience}',flush=True)
        if bad>=args.patience:
            break
    if not best.exists():
        raise RuntimeError('No best checkpoint produced')
    ck = torch.load(best, map_location='cpu', weights_only=True)
    report=dict(stage=7, split='validation_only',settings=settings,
                best_epoch=int(ck['epoch']),**ck['metrics'],
                checkpoint=str(best),
                caveat='Exploratory validation study; frozen seed42 encoders; no test evaluation')
    atomic_json(report, summary_path)
    atomic_json(report, done)
    print('COMPLETE', prefix,'best val AUPRC',round(report['val_auprc'],5),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())
