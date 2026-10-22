"""Resumable validation-only frozen-feature ablation runner; no test evaluation."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / '16_train_stage6_ablations.py'
DIAG = ROOT / 'scripts' / '15_stage6_diagnostics.py'
RESULTS = ROOT / 'results'
CKPT = ROOT / 'checkpoints'
VARIANTS = ('full', 'ehr_only', 'no_reliability', 'no_gate',
            'no_change', 'no_align_loss', 'no_stage3_init')


def stamp(msg):
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}', flush=True)


def call(parts):
    rc = subprocess.run([sys.executable, '-u', *map(str, parts)], cwd=ROOT).returncode
    if rc != 0:
        raise RuntimeError(f'Failed exit {rc}: {parts}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    variants = [v.strip() for v in args.variants.split(',')]
    if not seeds or any(v not in VARIANTS for v in variants):
        parser.error('Invalid seed or variant')

    stamp('Start validation-only Stage 6A diagnostics')
    call([DIAG])
    stamp('Smoke tests: ' + ', '.join(variants))
    for variant in variants:
        call([SCRIPT, '--variant', variant, '--seed', seeds[0], '--smoke'])
    if args.smoke_only:
        stamp('SMOKE TESTS COMPLETE; no ablation runs started')
        return

    for seed in seeds:
        for variant in variants:
            prefix = f'trcg_stage6b_{variant}_seed{seed}'
            # Child process validates stored settings even for completed runs.
            stamp('Starting: ' + prefix)
            call([SCRIPT, '--variant', variant, '--seed', seed, '--resume'])

    reports = []
    for seed in seeds:
        for variant in variants:
            filename = RESULTS / f'trcg_stage6b_{variant}_seed{seed}_summary.json'
            if not filename.exists():
                raise FileNotFoundError(filename)
            info = json.loads(filename.read_text(encoding='utf-8'))
            reports.append({'seed': seed, 'variant': variant,
                            'best_epoch': info['best_epoch'],
                            'val_auroc': info['val_auroc'],
                            'val_auprc': info['val_auprc']})
    table = pd.DataFrame(reports)
    table.to_csv(RESULTS / 'trcg_stage6b_all_validation_runs.csv', index=False)
    result = table.groupby('variant').agg(
        runs=('seed', 'count'),
        val_auprc_mean=('val_auprc', 'mean'),
        val_auprc_std=('val_auprc', 'std'),
        val_auroc_mean=('val_auroc', 'mean')
    ).reset_index().sort_values('val_auprc_mean', ascending=False)
    result.to_csv(RESULTS / 'trcg_stage6b_validation_comparison.csv', index=False)
    print('\nSTAGE 6B VALIDATION RESULTS (all frozen Stage-2 encoders):')
    print(result.to_string(index=False, float_format=lambda a: f'{a:.4f}'))
    stamp('COMPLETE - all experiments validation-only; no test metrics')


if __name__ == '__main__':
    main()
