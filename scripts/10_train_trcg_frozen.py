"""Stage-4A: two comparable frozen-feature TRCG fusion ablations.

Run commands:
  python scripts/10_train_trcg_frozen.py prepare
  python scripts/10_train_trcg_frozen.py smoke
  python scripts/10_train_trcg_frozen.py train --variant v1 --resume
  python scripts/10_train_trcg_frozen.py train --variant v2 --resume
  python scripts/10_train_trcg_frozen.py select-test

v1 = randomly initialized cross-modal projection heads
v2 = Stage-3 initialized cross-modal projection heads
Both use the same architecture, loss and cached Stage-2 encoder features.
"""
import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.trcg_fusion import TRCGFusion

CACHE, CKPT, RESULT, LOG = (ROOT / x for x in ('cache', 'checkpoints', 'results', 'logs'))
for p in (CACHE, CKPT, RESULT, LOG):
    p.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

S2E = CKPT / 'trcg_stage2_ehr_seed42_best.pt'
S2C = CKPT / 'trcg_stage2_cxr_seed42_best.pt'
S3 = CKPT / 'trcg_stage3_align_seed42_best.pt'
INPUT_KEYS = ('ehr_at_48', 'ehr_at_cxr', 'cxr', 'cxr_age',
              'anchor_valid', 'ehr_quality', 'cxr_entropy')


def atomic_torch_save(data, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temp)
    os.replace(temp, path)


def atomic_npz(path, **data):
    temp = path.with_suffix('.npz.tmp')
    with open(temp, 'wb') as f:
        np.savez_compressed(f, **data)
    os.replace(temp, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare():
    """Join Stage-3 features to event quality (by stay_id), never by row order."""
    manifest = pd.read_csv(CACHE / 'trcg_manifest.csv')
    for checkpoint in (S2E, S2C):
        if not checkpoint.exists():
            raise FileNotFoundError(f'Missing Stage-2 checkpoint: {checkpoint}')
    cxr_state = torch.load(S2C, map_location='cpu', weights_only=True)['model']
    w = cxr_state['head.weight'].float().numpy().reshape(-1)
    bias = float(cxr_state['head.bias'].item())
    if w.shape != (512,):
        raise ValueError('Unexpected frozen CXR feature/head dimensions')
    for split in ('train', 'val', 'test'):
        source = CACHE / f'trcg_stage3_{split}_features.npz'
        if not source.exists():
            if split != 'test':
                raise FileNotFoundError(f'Stage-3 {split} cache missing: {source}')
            print('Caching Stage-3 TEST encoder features once (no training)...', flush=True)
            path = ROOT / 'scripts' / '08_train_temporal_alignment.py'
            spec = importlib.util.spec_from_file_location('stage3_temporal_alignment', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.extract_split('test', batch_size=4)
        out = CACHE / f'trcg_stage4_{split}_features.npz'
        if out.exists():
            print('Reuse:', out, flush=True)
            continue
        with np.load(source) as source_npz:
            f = {key: source_npz[key] for key in source_npz.files}
        ids = np.asarray([str(s) for s in f['stay_id']])
        if len(set(ids)) != len(ids):
            raise RuntimeError(f'Duplicate stay IDs in {source}')
        events = torch.load(CACHE / f'{split}_event_cache.pt',
                            map_location='cpu', weights_only=False)  # local trusted cache
        by_id = {str(e['stay_id']): e for e in events}
        labels = dict(zip(manifest.loc[manifest.split == split, 'stay_id'].astype(str),
                          manifest.loc[manifest.split == split, 'label'].astype(int)))
        quality = []
        for i, stay in enumerate(ids):
            if stay not in by_id or stay not in labels:
                raise RuntimeError(f'Missing EHR data or manifest ID {stay}')
            if int(f['label'][i]) != labels[stay]:
                raise RuntimeError(f'Label mismatch for stay {stay}')
            ev = by_id[stay]
            m = ev['mask'].float().numpy()
            times = ev['times'].float().numpy()
            if len(times) == 0 or m.shape[1] != 17:
                raise RuntimeError(f'Invalid EHR sequence for stay {stay}')
            counts = int(m.sum())
            ever = m.max(axis=0).astype(bool)
            last_ages = []
            for j in range(17):
                observed = times[m[:, j] > .5]
                last_ages.append(48 - observed[-1] if len(observed) else 48.)
            gaps = np.diff(np.r_[0., times, 48.])
            quality.append([
                np.log1p(counts) / np.log1p(1500),
                1.0 - float(ever.mean()),
                float(np.median(last_ages)) / 48.,
                float(gaps.max()) / 48.,
                float(m.mean()),
            ])
        zc = f['cxr'].astype(np.float32)
        logits = np.clip(zc @ w + bias, -35., 35.)
        prob = 1 / (1 + np.exp(-logits))
        eps = 1e-7
        entropy = -(prob * np.log(prob + eps) +
                    (1 - prob) * np.log(1 - prob + eps)) / np.log(2.)
        result = {
            'ehr_at_48': f['ehr_at_48'].astype(np.float32),
            'ehr_at_cxr': f['ehr_at_cxr'].astype(np.float32),
            'cxr': zc,
            'cxr_age': f['cxr_age'].astype(np.float32),
            'anchor_valid': f['anchor_valid'].astype(np.float32),
            'ehr_quality': np.asarray(quality, dtype=np.float32),
            'cxr_entropy': entropy.astype(np.float32),
            'label': f['label'].astype(np.float32),
            'stay_id': ids,
        }
        n = len(ids)
        expected = len(manifest.loc[manifest.split == split])
        if n != expected:
            raise RuntimeError(f'{split}: {n} vs {expected} rows')
        if not all(np.isfinite(result[k]).all() for k in (*INPUT_KEYS, 'label')):
            raise RuntimeError(f'{split}: non-finite features')
        if (result['cxr_age'] < -1e-4).any() or (result['cxr_age'] > 48.001).any():
            raise RuntimeError(f'{split}: invalid CXR ages')
        atomic_npz(out, **result)
        print(f'Cached {split}: {n} patients, {int(result["anchor_valid"].sum())} valid anchors: {out}', flush=True)


class FeatureDataset(Dataset):
    def __init__(self, split):
        p = CACHE / f'trcg_stage4_{split}_features.npz'
        if not p.exists():
            raise FileNotFoundError(f'Missing {p}; run prepare')
        with np.load(p) as data:
            self.x = {key: torch.tensor(data[key], dtype=torch.float32)
                      for key in INPUT_KEYS}
            self.y = torch.tensor(data['label'], dtype=torch.float32)
            self.ids = [str(i) for i in data['stay_id']]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.x.items()}, self.y[i], self.ids[i]


def make_model(variant):
    es = torch.load(S2E, map_location='cpu', weights_only=True)['model']
    cs = torch.load(S2C, map_location='cpu', weights_only=True)['model']
    ehr_head = {'weight': es['head.weight'], 'bias': es['head.bias']}
    cxr_head = {'weight': cs['head.weight'], 'bias': cs['head.bias']}
    align_state = None
    if variant == 'v2':
        if not S3.exists():
            raise FileNotFoundError(f'No Stage-3 projection checkpoint: {S3}')
        align_state = torch.load(S3, map_location='cpu', weights_only=True)['model']
    return TRCGFusion(ehr_head, cxr_head, align_state).to(DEVICE)


def supervised_loss(out, batch, y, pos_weight, lambda_uni, lambda_align, lambda_comp):
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mort = bce(out['logits'], y)
    uni = bce(out['ehr_logits'], y) + 0.25 * bce(out['cxr_logits'], y)
    valid = batch['anchor_valid'].bool()
    alignment = mort.new_zeros(())
    if lambda_align > 0 and valid.sum().item() >= 2:
        e = out['q_ehr'][valid]
        c = out['q_cxr'][valid]
        targets = torch.arange(e.shape[0], device=e.device)
        logits = (e @ c.T) / 0.07
        alignment = 0.5 * (F.cross_entropy(logits, targets) +
                           F.cross_entropy(logits.T, targets))
    # Optional weak penalty for an old image after substantial clinical change.
    with torch.no_grad():
        change = (batch['ehr_at_48'] - batch['ehr_at_cxr']).norm(dim=-1)
        change = (change / (change + 3.)).clamp(0, 1)
        age = (batch['cxr_age'] / 48.).clamp(0, 1)
        penalty_weight = age * change * batch['anchor_valid']
    comp = (out['gamma'] * penalty_weight).mean()
    total = mort + lambda_uni * uni + lambda_align * alignment + lambda_comp * comp
    return total, mort, uni, alignment, comp


@torch.no_grad()
def evaluate(model, loader, pos_weight):
    model.eval()
    ids, y_all, p_all, gamma, r_ehr, r_cxr = [], [], [], [], [], []
    losses = []
    for x, y, patient_ids in loader:
        x = {k: v.to(DEVICE) for k, v in x.items()}
        y = y.to(DEVICE)
        out = model(x)
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(out['logits'], y)
        losses.append(float(loss))
        ids.extend(patient_ids)
        y_all.extend(y.cpu().numpy())
        p_all.extend(torch.sigmoid(out['logits']).cpu().numpy())
        gamma.extend(out['gamma'].cpu().numpy())
        r_ehr.extend(out['r_ehr'].cpu().numpy())
        r_cxr.extend(out['r_cxr'].cpu().numpy())
    return {
        'loss': float(np.mean(losses)),
        'auroc': float(roc_auc_score(y_all, p_all)),
        'auprc': float(average_precision_score(y_all, p_all)),
        'stay_id': ids,
        'labels': y_all, 'probabilities': p_all,
        'gamma': gamma, 'r_ehr': r_ehr, 'r_cxr': r_cxr,
    }


def get_prefix(variant, seed):
    return f'trcg_stage4a_{variant}_seed{seed}'


def train(args):
    seed_everything(args.seed)
    train_ds, val_ds = FeatureDataset('train'), FeatureDataset('val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    n_positive = float(train_ds.y.sum().item())
    pos_weight = torch.tensor((len(train_ds) - n_positive) / n_positive, device=DEVICE)
    model = make_model(args.variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    prefix = get_prefix(args.variant, args.seed)
    best, last = CKPT / (prefix + '_best.pt'), CKPT / (prefix + '_last.pt')
    history_path = LOG / (prefix + '_history.csv')
    summary_path = RESULT / (prefix + '_summary.json')
    done = CKPT / (prefix + '.done')
    start, best_ap, best_epoch, bad = 1, -1., 0, 0
    if args.resume and last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start, best_ap, best_epoch, bad = ck['epoch'] + 1, ck['best_ap'], ck['best_epoch'], ck['bad']
        print(f'Resuming {args.variant} from epoch {start}, best={best_epoch}, bad={bad}', flush=True)
    if args.smoke:
        x, y, _ = next(iter(train_loader))
        x = {k: v.to(DEVICE) for k, v in x.items()}
        y = y.to(DEVICE)
        out = model(x)
        loss, *_ = supervised_loss(out, x, y, pos_weight,
                                    args.lambda_uni, args.lambda_align, args.lambda_comp)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite smoke loss')
        print(f'SMOKE PASSED {args.variant} loss={loss.item():.4f} '
              f'gamma={out["gamma"].mean().item():.3f}', flush=True)
        return
    for epoch in range(start, args.epochs + 1):
        if bad >= args.patience:
            print('Patience exhausted before restart; evaluating best', flush=True)
            break
        model.train()
        total_losses = []
        for x, y, _ in train_loader:
            x = {k: v.to(DEVICE) for k, v in x.items()}
            y = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss, *_ = supervised_loss(out, x, y, pos_weight,
                                        args.lambda_uni, args.lambda_align, args.lambda_comp)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_losses.append(float(loss.item()))
        val = evaluate(model, val_loader, pos_weight)
        if val['auprc'] > best_ap + 1e-6:
            best_ap, best_epoch, bad = val['auprc'], epoch, 0
            atomic_torch_save({'model': model.state_dict(), 'epoch': epoch,
                               'val_auroc': val['auroc'], 'val_auprc': val['auprc']}, best)
        else:
            bad += 1
        atomic_torch_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                           'epoch': epoch, 'best_ap': best_ap, 'best_epoch': best_epoch,
                           'bad': bad}, last)
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_auroc', 'val_auprc', 'best_epoch']
        with open(history_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(dict(epoch=epoch, train_loss=np.mean(total_losses),
                                 val_loss=val['loss'], val_auroc=val['auroc'],
                                 val_auprc=val['auprc'], best_epoch=best_epoch))
        print(f'{args.variant} epoch {epoch}: val AUROC={val["auroc"]:.4f} '
              f'AUPRC={val["auprc"]:.4f} best={best_epoch} patience={bad}/{args.patience}',
              flush=True)
        if bad >= args.patience:
            break
    if not best.exists():
        raise RuntimeError(f'No best Stage-4A checkpoint for {args.variant}')
    state = torch.load(best, map_location='cpu', weights_only=True)
    summary = {
        'stage': '4A', 'variant': args.variant, 'seed': args.seed,
        'encoders_frozen': True, 'fusion_head_trainable': True,
        'best_epoch': int(state['epoch']),
        'val_auroc': float(state['val_auroc']),
        'val_auprc': float(state['val_auprc']),
        'train_examples': len(train_ds), 'val_examples': len(val_ds),
        'best_checkpoint': str(best),
        'stage3_alignment_initialization': args.variant == 'v2',
        'loss_weights': {'uni': args.lambda_uni, 'align': args.lambda_align,
                         'comp': args.lambda_comp},
        'note': 'Stage-4A frozen encoder feature ablation; not end-to-end Stage-4B.',
    }
    tmp = summary_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    os.replace(tmp, summary_path)
    done.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'STAGE 4A {args.variant} DONE: val AUPRC={summary["val_auprc"]:.4f}', flush=True)


def select_and_test(seed):
    summaries = []
    for variant in ('v1', 'v2'):
        path = RESULT / (get_prefix(variant, seed) + '_summary.json')
        if not path.exists():
            raise FileNotFoundError(f'Variant {variant} incomplete: {path}')
        summaries.append(json.loads(path.read_text(encoding='utf-8')))
    # VALIDATION selects method. No test-driven choice.
    chosen = sorted(summaries, key=lambda x: (-x['val_auprc'], -x['val_auroc'], x['variant']))[0]
    out_path = RESULT / f'trcg_stage4a_selected_seed{seed}_summary.json'
    if out_path.exists():
        print('Previously selected and tested:', out_path)
        return
    print('Selected via VAL AUPRC:', chosen['variant'], chosen['val_auprc'], flush=True)
    ds = FeatureDataset('test')
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    train_ds = FeatureDataset('train')
    positive = float(train_ds.y.sum().item())
    pos_weight = torch.tensor((len(train_ds) - positive) / positive, device=DEVICE)
    model = make_model(chosen['variant'])
    state = torch.load(chosen['best_checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    test = evaluate(model, loader, pos_weight)
    pred_path = RESULT / f'trcg_stage4a_selected_seed{seed}_test_predictions.csv'
    pd.DataFrame({
        'stay_id': test['stay_id'], 'label': test['labels'],
        'probability': test['probabilities'], 'gamma': test['gamma'],
        'r_ehr': test['r_ehr'], 'r_cxr': test['r_cxr'],
    }).to_csv(pred_path, index=False)
    result = {
        'stage': '4A', 'selection_rule': 'best validation AUPRC, then AUROC',
        'selected_variant': chosen['variant'],
        'comparison': [{'variant': x['variant'], 'val_auroc': x['val_auroc'],
                        'val_auprc': x['val_auprc']} for x in summaries],
        'test_auroc': test['auroc'], 'test_auprc': test['auprc'],
        'best_checkpoint': chosen['best_checkpoint'],
        'predictions_file': str(pred_path),
        'note': 'Frozen encoder TRCG proof-of-concept; encoder fine-tuning and calibration remain.',
    }
    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2), encoding='utf-8')
    os.replace(tmp, out_path)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'smoke', 'train', 'select-test'))
    parser.add_argument('--variant', choices=('v1', 'v2'), default='v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda-uni', type=float, default=0.2)
    parser.add_argument('--lambda-align', type=float, default=0.02)
    parser.add_argument('--lambda-comp', type=float, default=0.0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command in ('train', 'smoke'):
        args.smoke = args.command == 'smoke'
        train(args)
    else:
        select_and_test(args.seed)


if __name__ == '__main__':
    main()
