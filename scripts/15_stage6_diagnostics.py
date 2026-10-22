"""Stage 6A: validation-only TRCG behavior diagnostics (no training/test access).

Run: python scripts/15_stage6_diagnostics.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
CACHE = ROOT / 'cache'


def finite_corr(a, b):
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-12 or np.std(y[valid]) < 1e-12:
        return None
    val = spearmanr(x[valid], y[valid]).statistic
    return None if not np.isfinite(val) else float(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    val_path = RESULTS / 'trcg_stage5_val_calibrated_seed42.csv'
    cache_path = CACHE / 'trcg_stage4_val_features.npz'
    if not val_path.exists() or not cache_path.exists():
        raise FileNotFoundError('Require Stage 5 validation predictions and Stage 4 validation cache')
    val = pd.read_csv(val_path, dtype={'stay_id': str})
    needed = ('stay_id', 'label', 'probability', 'ehr_probability',
              'gamma', 'r_ehr', 'r_cxr')
    if any(c not in val for c in needed):
        raise ValueError(f'Stage-5 validation CSV needs columns {needed}')
    if val.stay_id.duplicated().any():
        raise ValueError('Duplicate validation stay_id')
    with np.load(cache_path) as data:
        keys = ('stay_id', 'label', 'ehr_at_48', 'ehr_at_cxr',
                'cxr_age', 'anchor_valid', 'cxr_entropy')
        if any(key not in data.files for key in keys):
            raise ValueError(f'Stage 4 cache needs {keys}')
        source = pd.DataFrame({
            'stay_id': data['stay_id'].astype(str),
            'cached_label': data['label'].astype(int),
            'cxr_age_hours': data['cxr_age'].astype(float),
            'anchor_valid': data['anchor_valid'].astype(int),
            'cxr_entropy': data['cxr_entropy'].astype(float),
            'ehr_change_norm': np.linalg.norm(
                data['ehr_at_48'] - data['ehr_at_cxr'], axis=1).astype(float),
        })
    if source.stay_id.duplicated().any():
        raise ValueError('Duplicate cached validation stay_id')
    merged = val.merge(source, how='inner', on='stay_id', validate='one_to_one')
    if len(merged) != len(val) or len(merged) != len(source):
        raise ValueError('Missing validation stay IDs; refusing positional alignment')
    if not np.array_equal(merged.label.to_numpy().astype(int),
                          merged.cached_label.to_numpy().astype(int)):
        raise ValueError('Label mismatch between validation CSV and feature cache')
    merged['effective_cxr_weight'] = merged.gamma * merged.r_cxr
    merged['age_bin'] = pd.qcut(merged.cxr_age_hours, 4, duplicates='drop').astype(str)

    y = merged.label.to_numpy(dtype=int)
    pf = merged.probability.to_numpy(dtype=float)
    pe = merged.ehr_probability.to_numpy(dtype=float)
    rng = np.random.default_rng(args.seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if min(len(pos), len(neg)) < 2:
        raise ValueError('Both classes required')
    diffs = []
    for _ in range(args.bootstrap):
        idx = np.r_[rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)]
        diffs.append(average_precision_score(y[idx], pf[idx]) -
                     average_precision_score(y[idx], pe[idx]))
    diffs = np.asarray(diffs)
    age_stats = merged.groupby('age_bin', observed=True).agg(
        n=('stay_id', 'size'),
        age_mean_h=('cxr_age_hours', 'mean'),
        gamma_mean=('gamma', 'mean'),
        r_cxr_mean=('r_cxr', 'mean'),
        effective_weight_mean=('effective_cxr_weight', 'mean'),
    ).sort_values('age_mean_h').reset_index()
    report = {
        'stage': '6A', 'scope': 'validation_only',
        'n': len(merged), 'deaths': int(y.sum()),
        'fused_auroc': float(roc_auc_score(y, pf)),
        'fused_auprc': float(average_precision_score(y, pf)),
        'ehr_aux_auroc': float(roc_auc_score(y, pe)),
        'ehr_aux_auprc': float(average_precision_score(y, pe)),
        'fused_minus_ehr_auprc': float(average_precision_score(y, pf) -
                                       average_precision_score(y, pe)),
        'fused_minus_ehr_auprc_paired_bootstrap_95ci':
            list(map(float, np.quantile(diffs, [0.025, 0.975]))),
        'bootstrap_reps': args.bootstrap,
        'valid_temporal_anchors': int(merged.anchor_valid.sum()),
        'gamma_mean': float(merged.gamma.mean()),
        'r_cxr_mean': float(merged.r_cxr.mean()),
        'effective_cxr_weight_mean': float(merged.effective_cxr_weight.mean()),
        'spearman_gamma_vs_cxr_age': finite_corr(merged.gamma, merged.cxr_age_hours),
        'spearman_effective_weight_vs_cxr_age': finite_corr(
            merged.effective_cxr_weight, merged.cxr_age_hours),
        'spearman_effective_weight_vs_ehr_change': finite_corr(
            merged.effective_cxr_weight, merged.ehr_change_norm),
        'age_quartiles': age_stats.to_dict(orient='records'),
        'caveats': [
            'Correlations describe behavior and do not establish clinically valid reliability.',
            'Bootstrap CI is descriptive: validation was used for checkpoint selection.',
            'Auxiliary EHR head and fused output share a trained model, not separately trained branches.',
            'No test files are read.',
        ],
    }
    RESULTS.mkdir(exist_ok=True, parents=True)
    (RESULTS / 'trcg_stage6a_diagnostics_seed42.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    age_stats.to_csv(RESULTS / 'trcg_stage6a_age_quartiles_seed42.csv', index=False)
    print('STAGE 6A VALIDATION DIAGNOSTICS')
    print('Fused AUPRC:', round(report['fused_auprc'], 4),
          'EHR-head AUPRC:', round(report['ehr_aux_auprc'], 4))
    print('Fused - EHR AUPRC:', round(report['fused_minus_ehr_auprc'], 4),
          'bootstrap 95% CI:', report['fused_minus_ehr_auprc_paired_bootstrap_95ci'])
    print('Effective CXR weight vs image age Spearman:',
          report['spearman_effective_weight_vs_cxr_age'])
    print('Saved:', RESULTS / 'trcg_stage6a_diagnostics_seed42.json')


if __name__ == '__main__':
    main()
