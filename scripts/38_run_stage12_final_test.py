"""EXPLICIT final retrospective evaluation. Requires Stage12 validation selection done.
 python scripts/38_run_stage12_final_test.py
 Test cohort has been inspected earlier in this project; NOT a fresh blind test.
"""
import argparse
import subprocess
import sys
from datetime import datetime
from stage12_common import ROOT,CACHE,RESULTS,ALL_SEEDS,CKPT,paths,align_path,fusion_path,feature_path

def run(file,*args):
    command=[sys.executable,'-u',str(ROOT/'scripts'/file),*map(str,args)]
    print(datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),'RUN',file,*args,flush=True)
    outcome=subprocess.run(command,cwd=ROOT,check=False)
    if outcome.returncode:raise RuntimeError(f'Failed {file} {args}; rerun once fixed')

def main(args):
    locked=RESULTS/'trcg_stage12_locked_temperatures.json'
    summary=RESULTS/'trcg_stage12_validation_summary.json'
    if not locked.exists() or not summary.exists():
        raise RuntimeError('First finish 37_run_stage12.py: validation temperatures must be locked.')
    test_quality=CACHE/'trcg_stage4_test_features.npz'
    if not test_quality.exists():
        raise FileNotFoundError(f'Missing original EHR quality descriptors: {test_quality}\n'
            'Run your existing Stage4A feature-preparation step to create this file; '
            'do not infer substitute descriptors.')
    if not (CACHE/'test_event_cache.pt').exists():
        raise FileNotFoundError(CACHE/'test_event_cache.pt')
    for seed in ALL_SEEDS:
        for f in (paths('ehr',seed),paths('cxr',seed),align_path(seed),fusion_path(seed)):
            if not f.exists():raise FileNotFoundError(f)
    for seed in ALL_SEEDS:
        run('33_stage12_cache_features.py','--seed',seed,'--split','test','--batch-size',args.cxr_batch)
    run('36_stage12_finalize.py','--final-test')
    print('FINAL FIVE-SEED RETROSPECTIVE TEST REPORT:',
          RESULTS/'trcg_stage12_final_test_summary.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cxr-batch',type=int,default=2)
    main(p.parse_args())
