"""Stage 9: READ-ONLY paired validation contribution analysis for TRCG.

Uses exact Stage 6 EHR-only, Stage 7 safe_age and Stage 8 dense_fixed
checkpoints for fusion seeds 42,43,44. All image/EHR encoders stay frozen;
there is no optimizer, gradient calculation, image loading, or test-set access.

Commands:
    python scripts/25_stage9_contribution_analysis.py --smoke
    python scripts/25_stage9_contribution_analysis.py
    python scripts/25_stage9_contribution_analysis.py --bootstrap 2000

Run from C:\\MedModWork\\TRCGFusion using the existing MedMod venv.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE, CHECKPOINTS, RESULTS = [ROOT / p for p in ('cache', 'checkpoints', 'results')]
SEEDS = (42, 43, 44)
FIELDS = ('ehr_at_48', 'ehr_at_cxr', 'cxr', 'cxr_age',
          'anchor_valid', 'ehr_quality', 'cxr_entropy')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_stage_module(number, filename):
    file = ROOT / 'scripts' / filename
    if not file.exists():
        raise FileNotFoundError(f'Missing prior-stage script: {file}')
    spec = importlib.util.spec_from_file_location(f'trcg_stage9_import_{number}', file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def fetch_npz(split, prefix):
    path = CACHE / f'{prefix}_{split}_features.npz'
    if not path.exists():
        raise FileNotFoundError(f'Missing feature cache {path}')
    with np.load(path, allow_pickle=False) as source:
        missing = set(FIELDS + ('stay_id', 'label')) - set(source.files)
        if missing:
            raise RuntimeError(f'{path}: missing {sorted(missing)}')
        ids = np.asarray(source['stay_id']).astype(str)
        labels = np.asarray(source['label'], dtype=np.int64)
        arrays = {k: np.asarray(source[k], dtype=np.float32) for k in FIELDS}
    if not len(ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f'{path}: zero rows or duplicate stays')
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError(f'{path}: labels not binary with both classes')
    if any(len(a) != len(ids) or not np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError(f'{path}: bad feature arrays')
    if len(labels) != len(ids):
        raise RuntimeError(f'{path}: label length mismatch')
    if np.any((arrays['cxr_age'] < -1e-4) | (arrays['cxr_age'] > 48.0001)):
        raise RuntimeError(f'{path}: out-of-window CXR ages')
    return ids, labels, arrays


def assemble_val_features():
    ids, y, stage7 = fetch_npz('val', 'trcg_stage4')
    ids8, y8, stage8 = fetch_npz('val', 'trcg_stage8')
    by8 = {patient: i for i, patient in enumerate(ids8)}
    if set(ids) != set(ids8):
        raise RuntimeError('Stage 7 and Stage 8 validation cohorts do not match by stay_id')
    reorder = np.array([by8[p] for p in ids], dtype=np.int64)
    if not np.array_equal(y, y8[reorder]):
        raise RuntimeError('Stage 7 and Stage 8 label mismatch after ID alignment')
    stage8 = {k: v[reorder] for k, v in stage8.items()}
    for k in ('ehr_at_48', 'ehr_at_cxr', 'cxr_age', 'anchor_valid', 'ehr_quality'):
        if not np.allclose(stage7[k], stage8[k], rtol=1e-4, atol=1e-4):
            raise RuntimeError(f'Stage 7/8 EHR or temporal feature mismatch: {k}')
    if stage7['ehr_at_48'].shape[1] != 128 or stage7['ehr_at_cxr'].shape[1] != 128:
        raise RuntimeError('Unexpected Stage 2 EHR feature dimensions')
    if stage7['cxr'].shape[1] != 512 or stage8['cxr'].shape[1] != 1024:
        raise RuntimeError('Unexpected ResNet/DenseNet feature dimensions')
    return ids, y, stage7, stage8


def batches(arrays, batch_size):
    n = len(next(iter(arrays.values())))
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = {
            k: torch.as_tensor(v[start:end], dtype=torch.float32, device=DEVICE)
            for k, v in arrays.items()
        }
        yield start, end, x


def restore_stage7(stage7_script, seed):
    path = CHECKPOINTS / f'trcg_stage7_safe_age_seed{seed}_best.pt'
    if not path.exists():
        raise FileNotFoundError(path)
    c = torch.load(path, map_location='cpu', weights_only=True)
    if c['settings']['variant'] != 'safe_age' or int(c['settings']['seed']) != seed:
        raise RuntimeError(f'Checkpoint metadata mismatch: {path}')
    ehr, cxr, align = stage7_script.source_states()
    gamma = float(c['settings']['gamma_from_stage4a_train'])
    model = stage7_script.SafeTRCG(ehr, cxr, align, 'safe_age', gamma)
    model.load_state_dict(c['model'], strict=True)
    model.to(DEVICE).eval()
    return model, path


def restore_stage8(stage8_script, seed):
    path = CHECKPOINTS / f'trcg_stage8_dense_fixed_seed{seed}_best.pt'
    if not path.exists():
        raise FileNotFoundError(path)
    c = torch.load(path, map_location='cpu', weights_only=True)
    if c['settings']['variant'] != 'dense_fixed' or int(c['settings']['seed']) != seed:
        raise RuntimeError(f'Checkpoint metadata mismatch: {path}')
    from src.models.trcg_densenet_fusion import DenseSafeTRCG
    ehr, cxr = stage8_script.source_states()
    gamma = float(c['settings']['fixed_gamma'])
    model = DenseSafeTRCG(ehr, cxr, 'dense_fixed', gamma)
    model.load_state_dict(c['model'], strict=True)
    model.to(DEVICE).eval()
    return model, path


def restore_stage6_ehr_only(seed):
    path = CHECKPOINTS / f'trcg_stage6b_ehr_only_seed{seed}_best.pt'
    if not path.exists():
        raise FileNotFoundError(path)
    c = torch.load(path, map_location='cpu', weights_only=True)
    if c['settings']['variant'] != 'ehr_only' or int(c['settings']['seed']) != seed:
        raise RuntimeError(f'Checkpoint metadata mismatch: {path}')
    state = c['model']
    # The Stage 6 EHR-only ablation is exactly its trained 128 -> 1 head.
    weight = state['ehr_head.weight'].detach().cpu().numpy().reshape(-1)
    bias = float(state['ehr_head.bias'].detach().cpu().item())
    if weight.shape != (128,):
        raise RuntimeError(f'Invalid trained Stage6 EHR head shape {path}')
    return weight, bias, path


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -70, 70)))


@torch.inference_mode()
def infer(model, features, batch_size):
    n = len(next(iter(features.values())))
    pred = {k: np.empty(n, dtype=np.float64) for k in (
        'fused_probability', 'ehr_probability', 'gamma', 'r_cxr',
        'effective_cxr_weight', 'signed_cxr_logit_shift')}
    for lo, hi, batch in batches(features, batch_size):
        out = model(batch)
        logit = out['logits'].reshape(-1).float().cpu().numpy()
        ehr = out['ehr_logits'].reshape(-1).float().cpu().numpy()
        pred['fused_probability'][lo:hi] = sigmoid(logit)
        pred['ehr_probability'][lo:hi] = sigmoid(ehr)
        pred['signed_cxr_logit_shift'][lo:hi] = logit - ehr
        pred['gamma'][lo:hi] = out['gamma'].reshape(-1).float().cpu().numpy()
        pred['r_cxr'][lo:hi] = out['r_cxr'].reshape(-1).float().cpu().numpy()
        pred['effective_cxr_weight'][lo:hi] = out['effective'].reshape(-1).float().cpu().numpy()
    if any(not np.isfinite(v).all() for v in pred.values()):
        raise RuntimeError('Non-finite predictions')
    return pred


def scores(y, probabilities):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1 - 1e-8)
    y = np.asarray(y, dtype=np.int64)
    if np.unique(y).size != 2:
        return dict(auprc=None, auroc=None, brier=float(np.mean((p - y) ** 2)),
                    log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return dict(auprc=float(average_precision_score(y, p)),
                auroc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, p, labels=(0, 1))))


def safe_spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    rho = float(spearmanr(x, y).statistic)
    return rho if np.isfinite(rho) else None


def metrics_frame(y, preds):
    out = []
    for key, pred in preds.items():
        out.append(dict(model=key, n=len(y), deaths=int(y.sum()), **scores(y, pred)))
    return pd.DataFrame(out)


def bootstrap(y, a, b, reps, seed):
    """Stratified paired bootstrap of validation AP and Brier differences.

    Uses the SAME indices for paired models and fixed validation class counts.
    Descriptive only: validation is already used for checkpoint/variant selection.
    """
    y = np.asarray(y, dtype=int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if min(len(pos), len(neg)) < 2:
        raise RuntimeError('Cannot bootstrap without both classes')
    obs_ap = scores(y, a)['auprc'] - scores(y, b)['auprc']
    obs_brier = scores(y, a)['brier'] - scores(y, b)['brier']
    rng = np.random.default_rng(seed)
    ap_deltas = np.empty(reps, dtype=float)
    brier_deltas = np.empty(reps, dtype=float)
    for k in range(reps):
        selected = np.r_[rng.choice(pos, len(pos), replace=True),
                         rng.choice(neg, len(neg), replace=True)]
        yy, aa, bb = y[selected], a[selected], b[selected]
        ap_deltas[k] = average_precision_score(yy, aa) - average_precision_score(yy, bb)
        brier_deltas[k] = np.mean((aa - yy) ** 2 - (bb - yy) ** 2)
    return dict(delta_auprc=float(obs_ap),
                delta_auprc_ci_low=float(np.quantile(ap_deltas, 0.025)),
                delta_auprc_ci_high=float(np.quantile(ap_deltas, 0.975)),
                delta_brier=float(obs_brier),
                delta_brier_ci_low=float(np.quantile(brier_deltas, 0.025)),
                delta_brier_ci_high=float(np.quantile(brier_deltas, 0.975)),
                bootstrap_replicates=int(reps))


def subgroup_table(df, split_key, candidates):
    rows = []
    for name, group in df.groupby(split_key, observed=True, sort=True):
        y = group['label'].to_numpy(dtype=int)
        age = group['cxr_age_hours'].to_numpy()
        for model in candidates:
            p = group[model].to_numpy(dtype=float)
            baseline = group['ehr_anchor'].to_numpy(dtype=float)
            metric = scores(y, p)
            base = scores(y, baseline)
            eps = 1e-8
            lp = -y * np.log(np.clip(p, eps, 1 - eps)) - (1-y)*np.log(np.clip(1-p, eps, 1-eps))
            lb = -y * np.log(np.clip(baseline, eps, 1 - eps)) - (1-y)*np.log(np.clip(1-baseline, eps, 1-eps))
            rows.append(dict(group_type=split_key, group=str(name), model=model,
                             n=len(group), deaths=int(y.sum()),
                             age_mean_h=float(age.mean()),
                             auprc=metric['auprc'], auroc=metric['auroc'],
                             brier=metric['brier'],
                             delta_auprc_vs_anchor=(None if metric['auprc'] is None or base['auprc'] is None
                                                    else float(metric['auprc']-base['auprc'])),
                             delta_brier_vs_anchor=float(metric['brier']-base['brier']),
                             fraction_individual_nll_improved=float(np.mean(lp < lb))))
    return pd.DataFrame(rows)


def run(args):
    # Never load train or test data. All trained checkpoints are read-only.
    ids, y, x7, x8 = assemble_val_features()
    if len(ids) != 490 or int(y.sum()) != 73:
        raise RuntimeError(f'Unexpected validation cohort: n={len(ids)}, deaths={int(y.sum())}')
    mod7 = load_stage_module(7, '18_train_stage7_refinement.py')
    mod8 = load_stage_module(8, '23_train_stage8_fusion.py')

    seeds = (42,) if args.smoke else SEEDS
    predictions = pd.DataFrame(dict(
        stay_id=ids,
        label=y,
        cxr_age_hours=x7['cxr_age'].reshape(-1).astype(float),
        anchor_valid=x7['anchor_valid'].reshape(-1).astype(int),
        ehr_change_norm=np.linalg.norm(x7['ehr_at_48']-x7['ehr_at_cxr'], axis=1),
    ))
    references = []
    for seed in seeds:
        print(f'Loading Stage 6 EHR-only seed={seed}', flush=True)
        weight, bias, path6 = restore_stage6_ehr_only(seed)
        predictions[f'ehr_stage6_seed{seed}'] = sigmoid(x7['ehr_at_48'] @ weight + bias)
        references.append(str(path6))

        print(f'Loading Stage 7 safe_age seed={seed}', flush=True)
        model7, path7 = restore_stage7(mod7, seed)
        o7 = infer(model7, x7, args.batch_size)
        del model7
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        for key, val in o7.items():
            predictions[f'stage7_{key}_seed{seed}'] = val
        references.append(str(path7))

        print(f'Loading Stage 8 dense_fixed seed={seed}', flush=True)
        model8, path8 = restore_stage8(mod8, seed)
        o8 = infer(model8, x8, args.batch_size)
        del model8
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        for key, val in o8.items():
            predictions[f'stage8_{key}_seed{seed}'] = val
        references.append(str(path8))
        if not np.allclose(o7['ehr_probability'], o8['ehr_probability'], atol=1e-5, rtol=1e-5):
            raise RuntimeError('Stage 7 and Stage 8 EHR anchors disagree')
        if seed == seeds[0]:
            ehr_anchor = o7['ehr_probability'].copy()
        elif not np.allclose(ehr_anchor, o7['ehr_probability'], atol=1e-5, rtol=1e-5):
            raise RuntimeError('Frozen EHR anchor predictions changed across fusion seeds')
        print(f'  Stage7 AP={scores(y, o7["fused_probability"])["auprc"]:.5f}, '
              f'Stage8 AP={scores(y, o8["fused_probability"])["auprc"]:.5f}', flush=True)

    if args.smoke:
        print(f'SMOKE PASSED: model loading, {len(ids)} aligned validation stays, '
              'CXR contribution and EHR anchor checks. No outputs written.', flush=True)
        return

    predictions['ehr_anchor'] = ehr_anchor
    predictions['ehr_stage6_mean'] = predictions[[f'ehr_stage6_seed{s}' for s in seeds]].mean(axis=1)
    for name in ('stage7', 'stage8'):
        predictions[f'{name}_fused_mean'] = predictions[
            [f'{name}_fused_probability_seed{s}' for s in seeds]].mean(axis=1)
        for field in ('gamma', 'r_cxr', 'effective_cxr_weight', 'signed_cxr_logit_shift'):
            predictions[f'{name}_{field}_mean'] = predictions[
                [f'{name}_{field}_seed{s}' for s in seeds]].mean(axis=1)

    candidates = dict(ehr_anchor=predictions['ehr_anchor'].to_numpy(),
                      ehr_stage6_mean=predictions['ehr_stage6_mean'].to_numpy(),
                      stage7_fused_mean=predictions['stage7_fused_mean'].to_numpy(),
                      stage8_fused_mean=predictions['stage8_fused_mean'].to_numpy())
    metrics = metrics_frame(y, candidates)
    pairs = [
        ('stage7_fused_mean', 'ehr_anchor'),
        ('stage8_fused_mean', 'ehr_anchor'),
        ('stage7_fused_mean', 'ehr_stage6_mean'),
        ('stage8_fused_mean', 'ehr_stage6_mean'),
        ('stage7_fused_mean', 'stage8_fused_mean'),
        ('ehr_stage6_mean', 'ehr_anchor'),
    ]
    comparisons = []
    for idx, (a, b) in enumerate(pairs):
        print(f'Paired bootstrap: {a} versus {b}', flush=True)
        comparisons.append(dict(model=a, comparator=b,
                                **bootstrap(y, candidates[a], candidates[b],
                                            args.bootstrap, args.seed + idx)))
    comparison_df = pd.DataFrame(comparisons)

    predictions['age_quartile'] = pd.qcut(
        predictions['cxr_age_hours'], 4, labels=('Q1 newest', 'Q2', 'Q3', 'Q4 oldest'))
    predictions['anchor_risk_quartile'] = pd.qcut(
        predictions['ehr_anchor'].rank(method='first'), 4,
        labels=('Q1 lowest', 'Q2', 'Q3', 'Q4 highest'))
    analyses = {}
    for name in ('stage7', 'stage8'):
        effect = predictions[f'{name}_signed_cxr_logit_shift_mean'].to_numpy()
        age = predictions['cxr_age_hours'].to_numpy()
        clinical_change = predictions['ehr_change_norm'].to_numpy()
        predictions[f'{name}_abs_logit_shift_mean'] = np.abs(effect)
        analyses[name] = dict(
            mean_gate=float(predictions[f'{name}_gamma_mean'].mean()),
            mean_reliability=float(predictions[f'{name}_r_cxr_mean'].mean()),
            mean_effective_gate=float(predictions[f'{name}_effective_cxr_weight_mean'].mean()),
            mean_abs_actual_logit_shift=float(np.abs(effect).mean()),
            median_abs_actual_logit_shift=float(np.median(np.abs(effect))),
            spearman_effective_gate_vs_age=safe_spearman(
                predictions[f'{name}_effective_cxr_weight_mean'], age),
            spearman_actual_abs_logit_shift_vs_age=safe_spearman(np.abs(effect), age),
            spearman_actual_abs_logit_shift_vs_ehr_change=safe_spearman(
                np.abs(effect), clinical_change),
            mean_signed_logit_shift=float(effect.mean()),
        )

    subgroup_candidates = ('ehr_anchor', 'ehr_stage6_mean', 'stage7_fused_mean', 'stage8_fused_mean')
    subgroups = pd.concat([
        subgroup_table(predictions, 'age_quartile', subgroup_candidates),
        subgroup_table(predictions, 'anchor_risk_quartile', subgroup_candidates),
    ], ignore_index=True)

    outputs = {
        'trcg_stage9_validation_predictions.csv': predictions,
        'trcg_stage9_model_comparison.csv': metrics,
        'trcg_stage9_paired_bootstrap.csv': comparison_df,
        'trcg_stage9_subgroup_analysis.csv': subgroups,
    }
    RESULTS.mkdir(exist_ok=True, parents=True)
    for name, frame in outputs.items():
        path = RESULTS / name
        frame.to_csv(path, index=False)
        print('SAVED', path, flush=True)
    summary = dict(
        stage=9, scope='validation_only',
        n=int(len(y)), deaths=int(y.sum()),
        fusion_seeds=list(seeds),
        seed_description='Only fusion heads trained independently; all seed42 EHR/CXR encoders frozen',
        sources=references,
        metrics=metrics.to_dict(orient='records'),
        paired_comparisons=comparisons,
        cxr_influence_diagnostics=analyses,
        caveats=[
            'Validation data were used to choose all three stages/variants; bootstrap intervals are descriptive, not independent confirmation.',
            'Seed-averaged prediction results are exploratory post hoc ensembles, not separately validated deployable ensemble models.',
            'EHR anchor and Stage6 separately trained EHR-only control are DIFFERENT baselines; do not conflate them.',
            'Age and EHR-risk subgroup comparisons are exploratory, often small, and may have unstable AUPRC.',
            'Correlation between gate magnitude and age does not prove temporal reliability or causal clinical benefit.',
            'No test set, test prediction files, optimizer, or retraining were used.',
        ],
    )
    (RESULTS / 'trcg_stage9_summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print('SAVED', RESULTS / 'trcg_stage9_summary.json', flush=True)
    print('\nSTAGE 9 COMPLETE — validation only\n', metrics.to_string(index=False), flush=True)
    print('\nPaired deltas:\n', comparison_df.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='load seed42 checkpoints and verify validation prediction invariants')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.bootstrap < 200:
        parser.error('--bootstrap must be >= 200')
    if args.batch_size < 1:
        parser.error('--batch-size must be >= 1')
    run(args)
