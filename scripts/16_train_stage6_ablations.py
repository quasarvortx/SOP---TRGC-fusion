"""Stage 6B: *validation-only* matched frozen-feature ablations.

Encoder features/pretraining are fixed (seed 42); seed variations below affect
fusion initialization and training only. Does not read/evaluate test set.

Run:
  python scripts/16_train_stage6_ablations.py --variant full --seed 42 --smoke
  python scripts/16_train_stage6_ablations.py --variant full --seed 42 --resume
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.trcg_fusion import TRCGFusion

CACHE, CKPT, LOGS, RESULTS = (ROOT / p for p in
                             ('cache', 'checkpoints', 'logs', 'results'))
for directory in (CKPT, LOGS, RESULTS):
    directory.mkdir(parents=True, exist_ok=True)

VARIANTS = ('full', 'ehr_only', 'no_reliability', 'no_gate',
            'no_change', 'no_align_loss', 'no_stage3_init')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(data, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def atomic_json(data, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(tmp, path)


class TRCGAblation(TRCGFusion):
    """Same learned modules as TRCGFusion; change only named pathway."""
    def __init__(self, ehr_head_state, cxr_head_state, align_state, variant):
        super().__init__(ehr_head_state, cxr_head_state,
                         None if variant == 'no_stage3_init' else align_state)
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        if variant == 'ehr_only':
            for n, p in self.named_parameters():
                p.requires_grad_(n.startswith('ehr_head.'))

    def forward(self, b):
        zt, za, zc = b['ehr_at_48'], b['ehr_at_cxr'], b['cxr']
        age = (b['cxr_age'].reshape(-1, 1) / 48.).clamp(0, 1)
        valid = b['anchor_valid'].float().reshape(-1, 1)
        entropy = b['cxr_entropy'].reshape(-1, 1)
        qe, qc = self.projections(za, zc)
        concordance = torch.cat((qc, qe, (qc-qe).abs(), qc*qe), dim=-1)
        if self.variant == 'no_change':
            d = torch.zeros_like(zt)
        else:
            d = self.change(torch.cat((zt, za, zt-za), dim=-1))
        if self.variant == 'no_reliability':
            re, rc = torch.ones_like(age), torch.ones_like(age)
        else:
            re = torch.sigmoid(self.ehr_reliability(b['ehr_quality']))
            rc = torch.sigmoid(self.cxr_reliability(
                torch.cat((age, entropy, self.image_quality(zc)), dim=-1)))
        delta = self.residual(torch.cat((zc, za, d, concordance), dim=-1))
        gate_in = torch.cat((concordance, d, age, re, rc, valid), dim=-1)
        if self.variant == 'no_gate':
            gamma = torch.ones_like(age)
        else:
            gamma = torch.sigmoid(self.gate(gate_in))
        fused = zt + gamma * rc * delta
        if self.variant == 'ehr_only':
            fused = zt
        return {
            'logits': self.ehr_head(fused).squeeze(-1),
            'ehr_logits': self.ehr_head(zt).squeeze(-1),
            'cxr_logits': self.cxr_head(zc).squeeze(-1),
            'gamma': gamma.squeeze(-1),
            'r_ehr': re.squeeze(-1),
            'r_cxr': rc.squeeze(-1),
            'q_ehr': qe, 'q_cxr': qc, 'change': d,
        }


def load_dataset():
    # Reuse the validated stage4 dataset class without modifying its code.
    from importlib.util import spec_from_file_location, module_from_spec
    path = ROOT / 'scripts' / '10_train_trcg_frozen.py'
    spec = spec_from_file_location('trcg_stage4a_import', path)
    mod = module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.FeatureDataset('train'), mod.FeatureDataset('val')


def new_model(variant):
    e = torch.load(CKPT / 'trcg_stage2_ehr_seed42_best.pt',
                   map_location='cpu', weights_only=True)['model']
    c = torch.load(CKPT / 'trcg_stage2_cxr_seed42_best.pt',
                   map_location='cpu', weights_only=True)['model']
    a = torch.load(CKPT / 'trcg_stage3_align_seed42_best.pt',
                   map_location='cpu', weights_only=True)['model']
    return TRCGAblation(
        {'weight': e['head.weight'], 'bias': e['head.bias']},
        {'weight': c['head.weight'], 'bias': c['head.bias']},
        a, variant).to(DEVICE)


def calc_loss(out, x, y, criterion, variant):
    mort = criterion(out['logits'], y)
    if variant == 'ehr_only':
        return mort
    uni = criterion(out['ehr_logits'], y) + 0.25*criterion(out['cxr_logits'], y)
    loss = mort + 0.2*uni
    if variant != 'no_align_loss':
        valid = x['anchor_valid'].bool()
        if valid.sum().item() >= 2:
            qe, qc = out['q_ehr'][valid], out['q_cxr'][valid]
            targets = torch.arange(len(qe), device=qe.device)
            logits = (qe @ qc.T)/0.07
            align = (F.cross_entropy(logits, targets) +
                     F.cross_entropy(logits.T, targets))/2
            loss = loss + 0.02*align
    return loss


@torch.inference_mode()
def validate(model, loader):
    model.eval()
    ys, probs = [], []
    for x, y, _ in loader:
        x = {k: v.to(DEVICE) for k, v in x.items()}
        out = model(x)
        ys.extend(y.numpy().tolist())
        probs.extend(torch.sigmoid(out['logits']).cpu().numpy().tolist())
    return (float(roc_auc_score(ys, probs)),
            float(average_precision_score(ys, probs)))


def write_history(path, history):
    fields = ['epoch', 'train_loss', 'val_auroc', 'val_auprc', 'best_epoch', 'bad']
    tmp = path.with_suffix('.csv.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(history)
    os.replace(tmp, path)


def run(args):
    seed_everything(args.seed)
    train_ds, val_ds = load_dataset()
    if min(len(train_ds), len(val_ds)) == 0:
        raise RuntimeError('Empty training or validation dataset')
    dl_tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=0)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)
    n_pos = float(train_ds.y.sum().item())
    if n_pos == 0 or n_pos >= len(train_ds):
        raise ValueError('Invalid mortality class balance')
    weight = torch.tensor((len(train_ds)-n_pos)/n_pos, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    model = new_model(args.variant)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                   lr=args.lr, weight_decay=1e-4)
    prefix = f'trcg_stage6b_{args.variant}_seed{args.seed}'
    best, last, done = [CKPT / (prefix + suffix) for suffix in
                        ('_best.pt', '_last.pt', '.done')]
    summary_path = RESULTS / (prefix + '_summary.json')
    history_path = LOGS / (prefix + '_history.csv')
    settings = {'variant': args.variant, 'seed': args.seed, 'lr': args.lr,
                'batch_size': args.batch_size, 'patience': args.patience,
                'epochs': args.epochs, 'lambda_uni': 0 if args.variant == 'ehr_only' else 0.2,
                'lambda_align': 0 if args.variant in ('ehr_only', 'no_align_loss') else 0.02,
                'stage3_init': args.variant != 'no_stage3_init'}
    if args.smoke:
        x, y, _ = next(iter(dl_tr))
        x = {k: v.to(DEVICE) for k, v in x.items()}
        out = model(x)
        loss = calc_loss(out, x, y.to(DEVICE), criterion, args.variant)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite smoke loss')
        loss.backward()
        print('SMOKE PASSED', args.variant, 'loss', round(loss.item(), 5), flush=True)
        return
    if done.exists():
        saved = json.loads(done.read_text(encoding='utf-8'))
        if saved['settings'] != settings:
            raise RuntimeError(f'{done} exists with different settings')
        print('ALREADY COMPLETE', prefix, flush=True)
        return

    epoch_start, best_ep, best_ap, best_auc, bad, history = 1, 0, -1., -1., 0, []
    if args.resume and last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['settings'] != settings:
            raise RuntimeError(f'Checkpoint {last} has different settings')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        epoch_start = state['epoch'] + 1
        best_ep, best_ap, best_auc, bad = (state[k] for k in
                                            ('best_epoch','best_ap','best_auc','bad'))
        history = state['history']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.random.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available() and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        write_history(history_path, history)
        print('RESUME', prefix, 'epoch', epoch_start, flush=True)

    for epoch in range(epoch_start, args.epochs + 1):
        if bad >= args.patience:
            break
        model.train()
        losses = []
        for x, y, _ in dl_tr:
            x = {k: v.to(DEVICE) for k, v in x.items()}
            y = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = calc_loss(out, x, y, criterion, args.variant)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        auc, ap = validate(model, dl_val)
        if ap > best_ap + 1e-6:
            best_ap, best_auc, best_ep, bad = ap, auc, epoch, 0
            atomic_save({'model': model.state_dict(), 'best_epoch': best_ep,
                         'val_auroc': best_auc, 'val_auprc': best_ap,
                         'settings': settings}, best)
        else:
            bad += 1
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                        'val_auroc': auc, 'val_auprc': ap,
                        'best_epoch': best_ep, 'bad': bad})
        atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'epoch': epoch, 'best_epoch': best_ep, 'best_ap': best_ap,
                     'best_auc': best_auc, 'bad': bad, 'history': history,
                     'settings': settings, 'python_rng': random.getstate(),
                     'numpy_rng': np.random.get_state(),
                     'torch_rng': torch.random.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                    last)
        write_history(history_path, history)
        print(f'{prefix}: epoch={epoch} AUROC={auc:.4f} AUPRC={ap:.4f} '
              f'best={best_ep} patience={bad}/{args.patience}', flush=True)
        if bad >= args.patience:
            break
    if not best.exists():
        raise RuntimeError(f'No best checkpoint: {best}')
    report = {'stage': '6B', 'split': 'validation', 'settings': settings,
              'best_epoch': best_ep, 'val_auroc': best_auc,
              'val_auprc': best_ap, 'checkpoint': str(best),
              'note': ('Only the frozen Stage-2/Stage-3 features are used; '
                       'multiple seeds vary fusion training only; test not read.')}
    atomic_json(report, summary_path)
    atomic_json(report, done)
    print('COMPLETE', prefix, 'best val AUPRC', round(best_ap, 5), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
