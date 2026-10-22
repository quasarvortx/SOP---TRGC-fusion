"""Resume-safe Stage 7 runner: three controlled variants, seeds 42/43/44."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'scripts' / '18_train_stage7_refinement.py'
RESULTS = ROOT / 'results'
CKPT = ROOT / 'checkpoints'
VARIANTS = ('safe_learned','safe_fixed','safe_age')
SEEDS = (42,43,44)


def summarize():
    rows=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            path=RESULTS / f'trcg_stage7_{variant}_seed{seed}_summary.json'
            if not path.exists():
                raise RuntimeError(f'Missing completed run: {path}')
            result=json.loads(path.read_text(encoding='utf-8'))
            rows.append(dict(variant=variant,seed=seed,best_epoch=result['best_epoch'],
                             val_auprc=result['val_auprc'],val_auroc=result['val_auroc'],
                             ehr_base_auprc=result['ehr_base_auprc'],
                             mean_abs_logit_shift=result['mean_abs_logit_shift'],
                             spearman_effective_vs_cxr_age=result['spearman_effective_vs_cxr_age']))
    path=RESULTS/'trcg_stage7_all_validation_runs.csv'
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    from statistics import mean, stdev
    summary=[]
    for variant in VARIANTS:
        group=[r for r in rows if r['variant']==variant]
        aps=[r['val_auprc'] for r in group]
        summary.append(dict(variant=variant,seeds=len(group),
            val_auprc_mean=mean(aps),val_auprc_sd=stdev(aps),
            val_auroc_mean=mean(r['val_auroc'] for r in group),
            ehr_base_auprc_mean=mean(r['ehr_base_auprc'] for r in group)))
    path=RESULTS/'trcg_stage7_validation_comparison.csv'
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print('STAGE 7 COMPLETE. Summary:',path,flush=True)
    for item in sorted(summary,key=lambda x:-x['val_auprc_mean']):
        print(item,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args()
    if args.smoke_only:
        for variant in VARIANTS:
            cmd=[sys.executable,'-u',str(TRAIN),'--variant',variant,'--smoke']
            subprocess.run(cmd,cwd=ROOT,check=True)
        print('ALL THREE SMOKE TESTS PASSED',flush=True)
        return
    for variant in VARIANTS:
        for seed in SEEDS:
            prefix=f'trcg_stage7_{variant}_seed{seed}'
            done=CKPT/f'{prefix}.done'
            if done.exists():
                print('SKIP completed',prefix,flush=True)
                continue
            cmd=[sys.executable,'-u',str(TRAIN),'--variant',variant,'--seed',str(seed),'--resume']
            print('START',prefix,flush=True)
            subprocess.run(cmd,cwd=ROOT,check=True)
            if not done.exists():
                raise RuntimeError(f'Run exited without completion: {prefix}')
    summarize()


if __name__=='__main__':
    main()
