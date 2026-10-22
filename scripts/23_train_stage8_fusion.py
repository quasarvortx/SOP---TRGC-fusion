"""Stage 8: train lightweight TRCG fusion with frozen DenseNet121 image features.

No test features or test labels accessed. Run after 22_cache_densenet_features.py.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.trcg_densenet_fusion import DenseSafeTRCG, VARIANTS

CACHE = ROOT / 'cache'
CKPT = ROOT / 'checkpoints'
RESULTS = ROOT / 'results'
LOGS = ROOT / 'logs'
for d in (CKPT, RESULTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUTS = ('ehr_at_48', 'ehr_at_cxr', 'cxr', 'cxr_age', 'anchor_valid',
          'ehr_quality', 'cxr_entropy')

class Cached(Dataset):
    def __init__(self, split):
        path = CACHE / f'trcg_stage8_{split}_features.npz'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}; run Stage8 cache script first')
        with np.load(path) as f:
            self.x = {k: torch.from_numpy(f[k].astype('float32')) for k in INPUTS}
            self.y = torch.from_numpy(f['label'].astype('float32'))
            self.ids = [str(v) for v in f['stay_id']]
        if len(self.y) != len(set(self.ids)) or len(self.y) == 0:
            raise RuntimeError(f'Empty or duplicate-ID data: {split}')
        for k, v in self.x.items():
            if len(v) != len(self.y) or not torch.isfinite(v).all():
                raise RuntimeError(f'Invalid {split} {k}')
        if (self.x['cxr'].shape[1], self.x['ehr_at_48'].shape[1], self.x['ehr_at_cxr'].shape[1]) != (1024, 128, 128):
            raise RuntimeError('Unexpected DenseNet/EHR feature dimensions')
        if self.x['ehr_quality'].shape[1] != 5:
            raise RuntimeError('Unexpected EHR quality feature dimension')
        if not ((self.y == 0) | (self.y == 1)).all():
            raise RuntimeError('Labels not binary')
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.x.items()}, self.y[i], self.ids[i]

def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

def atomic_json(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def atomic_csv(rows, path):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def source_states():
    es = torch.load(CKPT / 'trcg_stage2_ehr_seed42_best.pt', map_location='cpu', weights_only=True)['model']
    sd = torch.load(Path(r'C:\MedModWork\MedModRepo\checkpoints_local\densenet121_best.pt'),
                    map_location='cpu', weights_only=True)
    esh = {'weight': es['head.weight'], 'bias': es['head.bias']}
    csh = {'weight': sd['classifier.0.weight'], 'bias': sd['classifier.0.bias']}
    return esh, csh

def train_gamma():
    # Match Stage 7 fixed gate, derived exclusively from TRAIN.
    p = RESULTS / 'trcg_stage7_safe_learned_seed42_summary.json'
    if not p.exists():
        raise FileNotFoundError(f'Missing Stage7 seed42 summary for train-derived gamma: {p}')
    obj = json.loads(p.read_text(encoding='utf-8'))
    value = float(obj['settings']['gamma_from_stage4a_train'])
    if not (0 < value < 1):
        raise RuntimeError(f'Bad fixed gate {value}')
    return value

def objective(out, batch, y, weight, variant):
    mortality = nn.functional.binary_cross_entropy_with_logits(out['logits'], y, pos_weight=weight)
    penalty = mortality.new_zeros(())
    if variant == 'dense_age':
        with torch.no_grad():
            change = (batch['ehr_at_48'] - batch['ehr_at_cxr']).norm(dim=-1)
            change = change / (change + 3.)
            prior = (batch['cxr_age']/48.).clamp(0, 1) * change * batch['anchor_valid']
        penalty = (out['gamma'] * prior).mean()
    return mortality + (0.02 * penalty if variant == 'dense_age' else 0.)

@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    ys, probabilities, anchors, ages, influences = [], [], [], [], []
    for x, y, _ in loader:
        x = {k: v.to(DEVICE) for k,v in x.items()}
        o = model(x)
        ys.extend(y.cpu().tolist())
        probabilities.extend(torch.sigmoid(o['logits']).cpu().tolist())
        anchors.extend(torch.sigmoid(o['ehr_logits']).cpu().tolist())
        ages.extend(x['cxr_age'].cpu().flatten().tolist())
        influences.extend((o['logits'] - o['ehr_logits']).abs().cpu().tolist())
    return dict(val_auprc=float(average_precision_score(ys, probabilities)),
                val_auroc=float(roc_auc_score(ys, probabilities)),
                ehr_anchor_auprc=float(average_precision_score(ys, anchors)),
                ehr_anchor_auroc=float(roc_auc_score(ys, anchors)),
                mean_absolute_cxr_logit_correction=float(np.mean(influences)))

def train(args):
    seed_all(args.seed)
    train_ds, val_ds = Cached('train'), Cached('val')
    if set(train_ds.ids) & set(val_ds.ids):
        raise RuntimeError('Overlapping train/val stay IDs')
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    vloader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    es, cs = source_states()
    fixed = train_gamma()
    model = DenseSafeTRCG(es, cs, args.variant, fixed).to(DEVICE)
    positives = float(train_ds.y.sum())
    weight = torch.tensor((len(train_ds) - positives) / positives, device=DEVICE)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=1e-4)
    tag = f'trcg_stage8_{args.variant}_seed{args.seed}'
    best = CKPT / f'{tag}_best.pt'
    last = CKPT / f'{tag}_last.pt'
    done = CKPT / f'{tag}.done'
    history_path = LOGS / f'{tag}_history.csv'
    summary_path = RESULTS / f'{tag}_summary.json'
    settings = dict(stage=8, variant=args.variant, seed=args.seed,
                    epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                    lr=args.lr, fixed_gamma=round(fixed,8),
                    cxr_encoder='mortality_trained_densenet121_seed42',
                    ehr_encoder='trcg_stage2_seed42',
                    train_examples=len(train_ds), val_examples=len(val_ds))
    # Strict EHR-preserving initialization invariant.
    x0, _, _ = next(iter(vloader))
    x0 = {k: v.to(DEVICE) for k,v in x0.items()}
    model.eval()
    with torch.no_grad():
        o0 = model(x0)
    d = float((o0['logits'] - o0['ehr_logits']).abs().max())
    if d > 1e-5:
        raise RuntimeError(f'Initial EHR-anchor invariant failed: {d}')
    print('EHR anchor check PASSED; max initial difference:', d, flush=True)
    if args.smoke:
        model.train()
        x, y, _ = next(iter(loader))
        x = {k: v.to(DEVICE) for k, v in x.items()}
        loss = objective(model(x), x, y.to(DEVICE), weight, args.variant)
        if not torch.isfinite(loss):
            raise RuntimeError('Smoke loss not finite')
        loss.backward()
        print(f'SMOKE PASSED {args.variant}; loss={loss.item():.4f}', flush=True)
        return
    if done.exists():
        old = json.loads(done.read_text(encoding='utf-8'))
        if old['settings'] != settings:
            raise RuntimeError('Completed run uses other hyperparameters')
        print('ALREADY COMPLETE', tag)
        return
    start, top_ap, top_auc, best_epoch, bad, history = 1, -1., -1., 0, 0, []
    if args.resume and last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        if ck['settings'] != settings:
            raise RuntimeError(f'Cannot resume different settings: {last}')
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        start, top_ap, top_auc, best_epoch, bad, history = [ck[k] for k in
            ('epoch_next','top_ap','top_auc','best_epoch','bad','history')]
        torch.random.set_rng_state(ck['torch_rng'])
        random.setstate(ck['python_rng'])
        np.random.set_state(ck['numpy_rng'])
        if DEVICE.type == 'cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        print('RESUME', tag, 'epoch', start, flush=True)
    for ep in range(start, args.epochs+1):
        if bad >= args.patience:
            break
        model.train()
        epoch_loss = []
        for x, y, _ in loader:
            x = {k: v.to(DEVICE) for k,v in x.items()}
            y = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x), x, y, weight, args.variant)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss: {tag} epoch {ep}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))
        metrics = evaluate(model, vloader)
        better = (metrics['val_auprc'] > top_ap + 1e-6 or
                  (abs(metrics['val_auprc'] - top_ap) <= 1e-6 and metrics['val_auroc'] > top_auc + 1e-6))
        if better:
            top_ap, top_auc, best_epoch, bad = metrics['val_auprc'],metrics['val_auroc'],ep,0
            atomic_save(dict(model=model.state_dict(), epoch=ep, metrics=metrics, settings=settings), best)
        else:
            bad += 1
        history.append(dict(epoch=ep, train_loss=float(np.mean(epoch_loss)),
                            val_auprc=metrics['val_auprc'],val_auroc=metrics['val_auroc'],
                            ehr_anchor_auprc=metrics['ehr_anchor_auprc'],
                            mean_abs_cxr_logit_correction=metrics['mean_absolute_cxr_logit_correction']))
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         epoch_next=ep+1, top_ap=top_ap, top_auc=top_auc,
                         best_epoch=best_epoch, bad=bad, history=history,settings=settings,
                         torch_rng=torch.random.get_rng_state(), python_rng=random.getstate(),
                         numpy_rng=np.random.get_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else None),last)
        atomic_csv(history, history_path)
        print(f'{tag} epoch {ep}: val AUROC={metrics["val_auroc"]:.4f} '
              f'AUPRC={metrics["val_auprc"]:.4f} | EHR anchor={metrics["ehr_anchor_auprc"]:.4f} '
              f'best={best_epoch} bad={bad}/{args.patience}', flush=True)
        if bad >= args.patience:
            break
    if not best.exists():
        raise RuntimeError('No best checkpoint written')
    best_object = torch.load(best, map_location='cpu', weights_only=True)
    result = dict(stage=8, split='validation_only', settings=settings,
                  best_epoch=int(best_object['epoch']),
                  **best_object['metrics'], checkpoint=str(best),
                  caveat='Frozen seed42 encoders; contrast with Stage7 same cohort; no test evaluation')
    atomic_json(result, summary_path)
    atomic_json(result, done)
    print('COMPLETE', tag, 'best val AUPRC',round(result['val_auprc'],5), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    train(p.parse_args())
