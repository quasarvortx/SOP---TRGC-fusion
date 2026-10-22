"""Run Stage8 DenseNet feature cache + 9 frozen-feature controlled TRCG experiments.

python scripts/24_run_stage8.py --smoke-only
python scripts/24_run_stage8.py
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
RESULTS = ROOT / 'results'
CHECKPOINTS = ROOT / 'checkpoints'
VARIANTS = ('dense_learned','dense_fixed','dense_age')
SEEDS = (42,43,44)

def run(cmd):
    print('RUN', ' '.join(cmd), flush=True)
    subprocess.run([sys.executable, '-u', *cmd], cwd=ROOT, check=True)

def cache(args):
    for split in ('train','val'):
        cmd = [str(SCRIPTS/'22_cache_densenet_features.py'), '--split',split,
               '--batch-size',str(args.image_batch_size), '--transform',args.transform]
        if args.force_cache:
            cmd.append('--force')
        run(cmd)

def comparison():
    summaries=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            p = RESULTS / f'trcg_stage8_{variant}_seed{seed}_summary.json'
            if not p.exists():
                raise FileNotFoundError(p)
            o = json.loads(p.read_text(encoding='utf-8'))
            summaries.append(dict(variant=variant,seed=seed,best_epoch=o['best_epoch'],
                                  val_auprc=o['val_auprc'],val_auroc=o['val_auroc'],
                                  ehr_anchor_auprc=o['ehr_anchor_auprc']))
    path = RESULTS/'trcg_stage8_all_validation_runs.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    rows=[]
    for v in VARIANTS:
        sub=[s for s in summaries if s['variant']==v]
        ap=np.array([s['val_auprc'] for s in sub])
        auc=np.array([s['val_auroc'] for s in sub])
        rows.append(dict(variant=v,n=len(ap),mean_val_auprc=float(ap.mean()),
                         std_val_auprc=float(ap.std(ddof=1)),
                         mean_val_auroc=float(auc.mean()),std_val_auroc=float(auc.std(ddof=1))))
    out = RESULTS/'trcg_stage8_validation_comparison.csv'
    with out.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    winner=max(rows,key=lambda x:(x['mean_val_auprc'],x['mean_val_auroc']))
    report=dict(stage=8,selected_by='highest mean validation AUPRC over three fusion seeds',
                best_variant=winner['variant'],comparison=rows,
                previous_stage7_age_mean_ap=0.5498,
                previous_stage6_ehr_only_mean_ap=0.5511,
                caveat='Earlier controls are exploratory validation benchmarks; test not accessed.')
    (RESULTS/'trcg_stage8_validation_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nALL STAGE8 RUNS COMPLETED. Winner:',winner['variant'],
          'validation AUPRC=',round(winner['mean_val_auprc'],5), flush=True)
    print('Results:',out,flush=True)

def main(args):
    if not args.skip_cache:
        cache(args)
    for variant in VARIANTS:
        if args.smoke_only:
            run([str(SCRIPTS/'23_train_stage8_fusion.py'),
                 '--variant',variant,'--seed','42','--smoke'])
            continue
        for seed in SEEDS:
            print('\nSTART', variant, seed, flush=True)
            run([str(SCRIPTS/'23_train_stage8_fusion.py'),
                 '--variant',variant,'--seed',str(seed),'--resume'])
    if args.smoke_only:
        print('ALL STAGE8 SMOKE TESTS PASSED', flush=True)
    else:
        comparison()

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--skip-cache',action='store_true')
    p.add_argument('--force-cache',action='store_true')
    p.add_argument('--image-batch-size',type=int,default=2)
    p.add_argument('--transform',choices=('resize256_center224','resize224_square'),
                   default='resize256_center224')
    main(p.parse_args())
