"""Stage 10: CXR patient-pairing permutation controls (VALIDATION ONLY).

Run from TRCGFusion:
    python scripts/26_stage10_image_shuffle.py --smoke
    python scripts/26_stage10_image_shuffle.py --permutations 100

Depends on existing scripts/25_stage9_contribution_analysis.py and the
Stage 7/8 scripts, val feature caches, and best checkpoints. No training.
The experiment SHUFFLES CXR feature vectors and their derived uncertainty
jointly while keeping the recipient's EHR, CXR age, validity and quality fixed.
Within-age-quarter shuffles are also evaluated to control for age strata.
Do not interpret empirical shuffle tails as unbiased significance tests:
validation data were already used to select these checkpoints.
"""
from __future__ import annotations

import argparse
import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / 'results'


def load_stage9():
    path = ROOT / 'scripts' / '25_stage9_contribution_analysis.py'
    if not path.exists():
        raise FileNotFoundError(f'Stage 9 script required: {path}')
    name = 'trcg_stage10_stage9_helpers'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def score(y, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if len(p) != len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise RuntimeError('Invalid probability vector')
    return {
        'auprc': float(average_precision_score(y, p)),
        'auroc': float(roc_auc_score(y, p)),
        'brier': float(brier_score_loss(y, p)),
    }


def permutation(n, rng, age_groups=None):
    if age_groups is None:
        return rng.permutation(n)
    p = np.arange(n)
    for group in np.unique(age_groups):
        idx = np.flatnonzero(age_groups == group)
        if len(idx) > 1:
            p[idx] = rng.permutation(idx)
    return p


def substitute_cxr(features, p):
    """Intervene on image-derived channels only, preserving EHR and age."""
    result = dict(features)
    result['cxr'] = np.ascontiguousarray(features['cxr'][p])
    result['cxr_entropy'] = np.ascontiguousarray(features['cxr_entropy'][p])
    return result


def mean_probabilities(stage9, models, features, batch_size):
    preds = []
    for model in models:
        preds.append(stage9.infer(model, features, batch_size)['fused_probability'])
    return np.mean(np.stack(preds), axis=0)


def run(args):
    stage9 = load_stage9()
    ids, y, x7, x8 = stage9.assemble_val_features()
    if len(ids) != 490 or int(y.sum()) != 73:
        raise RuntimeError(f'Unexpected validation split: {len(ids)} / deaths={int(y.sum())}')

    stage7 = stage9.load_stage_module(7, '18_train_stage7_refinement.py')
    stage8 = stage9.load_stage_module(8, '23_train_stage8_fusion.py')
    age = x7['cxr_age'].reshape(-1)
    if not np.isfinite(age).all():
        raise RuntimeError('Nonfinite image age')
    # Equal-rank quartiles; deterministic, equal population up to 1.
    ranks = np.argsort(np.argsort(age, kind='stable'), kind='stable')
    age_bins = np.minimum(3, (4 * ranks // len(age)).astype(int))
    rng = np.random.default_rng(args.seed)
    models_to_use = [
        ('stage7_safe_age', x7, lambda s: stage9.restore_stage7(stage7, s)),
        ('stage8_dense_fixed', x8, lambda s: stage9.restore_stage8(stage8, s)),
    ]
    trials, comparisons, patient_rows = [], [], []
    n_reps = 1 if args.smoke else args.permutations
    seeds = (42,) if args.smoke else (42, 43, 44)

    for variant, features, restore in models_to_use:
        print(f'Loading {variant} (seeds {seeds})', flush=True)
        models = [restore(s)[0] for s in seeds]
        try:
            observed = mean_probabilities(stage9, models, features, args.batch_size)
            observed_m = score(y, observed)
            print(f'{variant} REAL: AP={observed_m["auprc"]:.6f}, '
                  f'AUROC={observed_m["auroc"]:.6f}, Brier={observed_m["brier"]:.6f}', flush=True)
            if args.smoke:
                p = permutation(len(y), rng)
                permuted = mean_probabilities(stage9, models, substitute_cxr(features, p), args.batch_size)
                if not np.isfinite(permuted).all() or len(permuted) != len(y):
                    raise RuntimeError('Smoke inference failed')
                print(f'SMOKE PASSED {variant}: real and shuffled predictions finite', flush=True)
                continue

            for mode in ('all_patients', 'same_age_quartile'):
                changed_predictions = []
                for i in range(n_reps):
                    p = permutation(len(y), rng, None if mode == 'all_patients' else age_bins)
                    altered_features = substitute_cxr(features, p)
                    shuffled = mean_probabilities(stage9, models, altered_features, args.batch_size)
                    sm = score(y, shuffled)
                    movement = np.abs(observed - shuffled)
                    trials.append(dict(variant=variant, shuffle_mode=mode, shuffle_seed=args.seed,
                                       trial=i, auprc=sm['auprc'], auroc=sm['auroc'],
                                       brier=sm['brier'], mean_abs_prediction_change=float(movement.mean()),
                                       max_abs_prediction_change=float(movement.max())))
                    changed_predictions.append(shuffled)
                    if (i + 1) % max(1, min(n_reps, 20)) == 0:
                        print(f'  {variant} {mode}: {i + 1}/{n_reps}', flush=True)

                trial_frame = pd.DataFrame([r for r in trials
                    if r['variant'] == variant and r['shuffle_mode'] == mode])
                # Image-pairing importance is not a formal generalization p-value.
                comparisons.append(dict(
                    variant=variant, shuffle_mode=mode, permutations=n_reps,
                    observed_auprc=observed_m['auprc'],
                    shuffled_auprc_mean=float(trial_frame.auprc.mean()),
                    shuffled_auprc_p025=float(trial_frame.auprc.quantile(.025)),
                    shuffled_auprc_p975=float(trial_frame.auprc.quantile(.975)),
                    observed_minus_shuffled_auprc=float(observed_m['auprc'] - trial_frame.auprc.mean()),
                    observed_brier=observed_m['brier'],
                    shuffled_brier_mean=float(trial_frame.brier.mean()),
                    observed_minus_shuffled_brier=float(observed_m['brier'] - trial_frame.brier.mean()),
                    mean_abs_prediction_change=float(trial_frame.mean_abs_prediction_change.mean()),
                    shuffled_fraction_at_least_observed_auprc=float(np.mean(trial_frame.auprc >= observed_m['auprc'])),
                ))
                stack = np.stack(changed_predictions)
                for j, patient_id in enumerate(ids):
                    patient_rows.append(dict(variant=variant, shuffle_mode=mode,
                                             stay_id=patient_id, label=int(y[j]),
                                             cxr_age_hours=float(age[j]),
                                             observed_probability=float(observed[j]),
                                             mean_shuffled_probability=float(stack[:, j].mean()),
                                             std_shuffled_probability=float(stack[:, j].std()),
                                             mean_abs_change=float(np.abs(stack[:, j]-observed[j]).mean())))
        finally:
            del models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.smoke:
        print('STAGE 10 SMOKE TESTS COMPLETE. No outputs written.', flush=True)
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    filenames = {
        'trcg_stage10_shuffle_trials.csv': pd.DataFrame(trials),
        'trcg_stage10_shuffle_comparison.csv': pd.DataFrame(comparisons),
        'trcg_stage10_patient_sensitivity.csv': pd.DataFrame(patient_rows),
    }
    for filename, frame in filenames.items():
        path = RESULTS / filename
        frame.to_csv(path, index=False)
        print('SAVED', path, flush=True)
    summary = {
        'stage': 10, 'scope': 'validation_only', 'n': len(y), 'deaths': int(y.sum()),
        'fusion_seeds': list(seeds), 'permutations_per_mode': n_reps,
        'shuffle_design': 'permuted CXR embedding and CXR-derived entropy jointly; patient EHR, CXR age, anchor validity, and EHR quality unchanged',
        'metrics': comparisons,
        'caveats': [
            'All comparisons are exploratory because validation was used to select checkpoints.',
            'Fusion seed variation is limited to fusion heads; encoders share original fixed seed.',
            'Image permutation is a negative control for patient-specific CXR signal, not proof of causal utility.',
            'Shuffling image embeddings may break conditional dependencies not captured by the age-quartile control.',
            'No test data, gradient updates, checkpoint writes, or optimizer states are used.',
        ],
    }
    out = RESULTS / 'trcg_stage10_shuffle_summary.json'
    out.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print('SAVED', out)
    print(pd.DataFrame(comparisons).to_string(index=False))
    print('STAGE 10 COMPLETE — validation only.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--permutations', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261020)
    a = parser.parse_args()
    if a.permutations < 1 or a.batch_size < 1:
        parser.error('permutations and batch-size must be positive')
    run(a)
