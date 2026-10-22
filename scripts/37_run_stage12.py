"""Unattended Stage12 training: new independent encoders seeds45/46, validation only.
 python scripts/37_run_stage12.py --smoke-only
 python scripts/37_run_stage12.py
 Repeat normal command to resume after restart.
"""
import argparse
import subprocess
import sys
from datetime import datetime
import pandas as pd
from stage12_common import ROOT,CACHE,CKPT,RESULTS,ALL_SEEDS,NEW_SEEDS,paths,align_path,fusion_path,feature_path

def info(*message):
    print(datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),*message,flush=True)

def call(file,*args):
    cmd=[sys.executable,'-u',str(ROOT/'scripts'/file),*map(str,args)]
    info('RUN',file,' '.join(map(str,args)))
    result=subprocess.run(cmd,cwd=ROOT,check=False)
    if result.returncode:
        raise RuntimeError(f'Failed {file} with exit {result.returncode}. Fix error and rerun runner; completed steps will be skipped.')

def preflight():
    req=[ROOT/'scripts/stage11_common.py',ROOT/'scripts/27_stage11_train_encoders.py',
         ROOT/'scripts/28_stage11_cache_features.py',ROOT/'scripts/29_stage11_train_alignment.py',
         ROOT/'scripts/30_stage11_train_fusion.py',ROOT/'scripts/06_train_unimodal.py',
         ROOT/'scripts/08_train_temporal_alignment.py',ROOT/'scripts/18_train_stage7_refinement.py',
         ROOT/'src/models/trcg_fusion.py',CACHE/'trcg_manifest.csv',CACHE/'ehr_preprocessor.json',
         CACHE/'train_event_cache.pt',CACHE/'val_event_cache.pt',
         CACHE/'trcg_stage4_train_features.npz',CACHE/'trcg_stage4_val_features.npz']
    for seed in (42,43,44):
        req.extend([paths('ehr',seed),paths('cxr',seed),align_path(seed),fusion_path(seed),
                    feature_path('train',seed),feature_path('val',seed),
                    RESULTS/f'trcg_stage11_safe_age_seed{seed}_val_predictions.csv'])
    missing=[str(x) for x in req if not x.is_file()]
    if missing:raise FileNotFoundError('Existing Stage11 inputs missing:\n'+'\n'.join(missing))
    manifest=pd.read_csv(CACHE/'trcg_manifest.csv',dtype={'stay_id':str,'subject_id':str})
    if manifest['stay_id'].duplicated().any():raise RuntimeError('Duplicate stay IDs')
    for split,count in (('train',4956),('val',490),('test',1231)):
        sub=manifest[manifest.split==split]
        if len(sub)!=count:raise RuntimeError(f'{split}: expected {count}, found {len(sub)}')
    subjects=[set(manifest.loc[manifest.split==s,'subject_id'].astype(str)) for s in ('train','val','test')]
    if any(subjects[i]&subjects[j] for i,j in ((0,1),(0,2),(1,2))):
        raise RuntimeError('Patient leakage across splits')
    info('PREFLIGHT PASS: original datasets and seeds42/43/44 verified')

def main(args):
    preflight()
    if args.smoke_only:
        call('32_stage12_train_encoders.py','--model','ehr','--seed',45,'--batch-size',4,'--smoke')
        call('32_stage12_train_encoders.py','--model','cxr','--seed',45,'--batch-size',args.cxr_batch,'--smoke')
        info('SMOKE TESTS PASSED for EHR and CXR. Start full Stage12.')
        return
    for seed in NEW_SEEDS:
        call('32_stage12_train_encoders.py','--model','ehr','--seed',seed,'--batch-size',8,'--resume')
        call('32_stage12_train_encoders.py','--model','cxr','--seed',seed,'--batch-size',args.cxr_batch,'--resume')
    for seed in NEW_SEEDS:
        for split in ('train','val'):
            call('33_stage12_cache_features.py','--seed',seed,'--split',split,
                 '--batch-size',args.cxr_batch)
        call('34_stage12_train_alignment.py','--seed',seed,'--smoke')
        call('34_stage12_train_alignment.py','--seed',seed,'--resume')
        call('35_stage12_train_fusion.py','--seed',seed,'--smoke')
        call('35_stage12_train_fusion.py','--seed',seed,'--resume')
    call('36_stage12_finalize.py','--val-only')
    info('STAGE12 TRAIN + VAL COMPLETE. Review results/trcg_stage12_validation_summary.json.')
    info('No test data used. For separate fixed-model test evaluation run 38_run_stage12_final_test.py')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--cxr-batch',type=int,default=2)
    main(p.parse_args())
