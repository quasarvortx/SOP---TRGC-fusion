"""Unattended Stage 11: independently trained 3-seed encoders + matched fusion.
Requires existing Stage 1-10 TRCGFusion files. Test set is never evaluated.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from stage11_common import (ROOT,CACHE,CKPT,LOGS,RESULTS,atomic_csv,atomic_json,
    read_feature,check_split_isolation)

SEEDS=(42,43,44)

def log(*msg):
    print(datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),*msg,flush=True)

def step(file,*args):
    cmd=[sys.executable,'-u',str(ROOT/'scripts'/file),*map(str,args)]
    log('RUN',' '.join(cmd[2:]))
    completed=subprocess.run(cmd,cwd=ROOT,check=False)
    if completed.returncode:
        raise RuntimeError(f'STAGE 11 HALTED (exit {completed.returncode}). '
                           f'Fix error above, then rerun this same command. Step: {file} {args}')

def preflight():
    required=[CACHE/'trcg_manifest.csv',CACHE/'ehr_preprocessor.json',
        CACHE/'train_event_cache.pt',CACHE/'val_event_cache.pt',
        CACHE/'trcg_stage4_train_features.npz',CACHE/'trcg_stage4_val_features.npz',
        CKPT/'trcg_stage2_ehr_seed42_best.pt',CKPT/'trcg_stage2_cxr_seed42_best.pt',
        CKPT/'trcg_stage3_align_seed42_best.pt',ROOT/'src/models/trcg_fusion.py',
        ROOT/'scripts/06_train_unimodal.py',ROOT/'scripts/08_train_temporal_alignment.py',
        ROOT/'scripts/18_train_stage7_refinement.py']
    missing=[str(p) for p in required if not p.exists()]
    if missing:raise FileNotFoundError('Missing dependencies:\n'+'\n'.join(missing))
    manifest=pd.read_csv(CACHE/'trcg_manifest.csv',dtype={'stay_id':str,'subject_id':str})
    train=manifest[manifest.split=='train'];val=manifest[manifest.split=='val']
    if len(train)!=4956 or len(val)!=490:
        raise RuntimeError(f'Unexpected cohort: train={len(train)} val={len(val)}. Audit first.')
    if train.stay_id.duplicated().any() or val.stay_id.duplicated().any():
        raise RuntimeError('Duplicate stay IDs')
    check_split_isolation({'stay_id':train.stay_id,'subject_id':train.subject_id},
                          {'stay_id':val.stay_id,'subject_id':val.subject_id})
    log('PREFLIGHT PASS: 4,956 train + 490 val; patient-level separation confirmed')

def summarize():
    perf=[];frames=[]
    for seed in SEEDS:
        path=RESULTS/f'trcg_stage11_safe_age_seed{seed}_val_predictions.csv'
        summary=RESULTS/f'trcg_stage11_safe_age_seed{seed}_summary.json'
        if not path.exists() or not summary.exists():raise FileNotFoundError(path)
        d=pd.read_csv(path,dtype={'stay_id':str,'subject_id':str})
        if d.stay_id.duplicated().any():raise RuntimeError('Duplicate predictions')
        if len(d)!=490:raise RuntimeError('Missing validation predictions')
        if not d[['fused_probability','ehr_probability']].apply(np.isfinite).all().all():
            raise RuntimeError('NaN validation predictions')
        y=d.label.to_numpy(dtype=int)
        if set(np.unique(y))!={0,1}:raise RuntimeError('Invalid validation labels')
        perf.append(dict(seed=seed,model='ehr_only',auprc=float(average_precision_score(y,d.ehr_probability)),
            auroc=float(roc_auc_score(y,d.ehr_probability)),brier=float(brier_score_loss(y,d.ehr_probability))))
        perf.append(dict(seed=seed,model='safe_age_fusion',auprc=float(average_precision_score(y,d.fused_probability)),
            auroc=float(roc_auc_score(y,d.fused_probability)),brier=float(brier_score_loss(y,d.fused_probability))))
        frames.append(d.set_index('stay_id')[['subject_id','label','ehr_probability','fused_probability']].rename(
           columns={'ehr_probability':f'ehr_{seed}','fused_probability':f'fusion_{seed}',
                    'subject_id':f'subject_{seed}','label':f'label_{seed}'}))
    allp=pd.concat(frames,axis=1,join='inner')
    if len(allp)!=490:raise RuntimeError('Seed prediction cohorts differ')
    for seed in SEEDS[1:]:
        if not allp[f'label_{seed}'].equals(allp['label_42']):raise RuntimeError('Mismatch labels')
        if not allp[f'subject_{seed}'].equals(allp['subject_42']):raise RuntimeError('Mismatch subjects')
    allp['label']=allp['label_42'];allp['subject_id']=allp['subject_42']
    allp['ehr_mean']=allp[[f'ehr_{s}' for s in SEEDS]].mean(axis=1)
    allp['fusion_mean']=allp[[f'fusion_{s}' for s in SEEDS]].mean(axis=1)
    atomic_csv(perf,RESULTS/'trcg_stage11_per_seed_validation.csv')
    flat=allp.reset_index().rename(columns={'index':'stay_id'})
    atomic_csv(flat.to_dict('records'),RESULTS/'trcg_stage11_ensemble_validation_predictions.csv')
    y=allp['label'].to_numpy(dtype=int)
    e=allp['ehr_mean'].to_numpy(dtype=float)
    f=allp['fusion_mean'].to_numpy(dtype=float)
    diff=float(average_precision_score(y,f)-average_precision_score(y,e))
    # Patient-cluster bootstrap (multiple stays can share one subject).
    ids=allp['subject_id'].astype(str).to_numpy()
    unique=np.unique(ids)
    by_subject={v:np.flatnonzero(ids==v) for v in unique}
    rng=np.random.default_rng(11042); boot=[]
    for _ in range(1000):
        draw=rng.choice(unique,size=len(unique),replace=True)
        idx=np.concatenate([by_subject[s] for s in draw])
        if len(set(y[idx]))<2:continue
        boot.append(average_precision_score(y[idx],f[idx])-average_precision_score(y[idx],e[idx]))
    if len(boot)<950:raise RuntimeError('Too few valid bootstrap samples')
    by_model={}
    for name in ('ehr_only','safe_age_fusion'):
        r=[x for x in perf if x['model']==name]
        by_model[name]={k+'_mean':float(np.mean([d[k] for d in r])) for k in ('auprc','auroc','brier')}
        by_model[name].update({k+'_std':float(np.std([d[k] for d in r],ddof=1)) for k in ('auprc','auroc','brier')})
    report=dict(stage=11,scope='validation_only',seed_list=list(SEEDS),
         encoder_training='Independent seeds 42,43,44 (same ImageNet ResNet initial weights, independently fine-tuned)',
         fusion_training='One separately trained safe_age head per independently trained EHR/CXR pair',
         per_seed=perf,mean_std=by_model,seed_ensemble=dict(
            ehr_auprc=float(average_precision_score(y,e)),
            fusion_auprc=float(average_precision_score(y,f)),
            ehr_auroc=float(roc_auc_score(y,e)),fusion_auroc=float(roc_auc_score(y,f)),
            ehr_brier=float(brier_score_loss(y,e)),fusion_brier=float(brier_score_loss(y,f)),
            delta_auprc=diff,
            patient_cluster_bootstrap_95ci=list(map(float,np.quantile(boot,[.025,.975]))),
            bootstrap_reps=len(boot)),
         limitations=[
          'Repeated use of validation for model development makes reported intervals exploratory.',
          'Same underlying train/validation split and shared ImageNet initialization; three independent downstream seeds are not three datasets.',
          'No calibration or final test evaluation performed in Stage11.',
          'EHR head frozen during safe residual fusion; EHR-only control is the exact paired unimodal head.',
          'Three seeds, not the five recommended for final deep ensemble.'
         ])
    atomic_json(report,RESULTS/'trcg_stage11_validation_summary.json')
    log('STAGE 11 COMPLETE!')
    log('Mean seed AUPRC: EHR',round(by_model['ehr_only']['auprc_mean'],4),
       'fusion',round(by_model['safe_age_fusion']['auprc_mean'],4))
    log('Ensemble EHR AP',round(report['seed_ensemble']['ehr_auprc'],4),
       'fusion',round(report['seed_ensemble']['fusion_auprc'],4))
    log('Saved',RESULTS/'trcg_stage11_validation_summary.json')

def main(args):
    preflight()
    if args.smoke_only:
        step('27_stage11_train_encoders.py','--model','ehr','--seed','43','--batch-size','4','--smoke')
        step('27_stage11_train_encoders.py','--model','cxr','--seed','43','--batch-size','2','--smoke')
        step('28_stage11_cache_features.py','--seed','42','--split','train')
        step('28_stage11_cache_features.py','--seed','42','--split','val')
        step('30_stage11_train_fusion.py','--seed','42','--smoke')
        log('STAGE 11 BASIC SMOKE TESTS PASSED. Can start unattended run.')
        return
    # Reuse the originally trained seed42 unimodal encoders and Stage3 projections.
    # Freshly train independently initialized unimodal encoders for 43/44.
    for seed in (43,44):
        step('27_stage11_train_encoders.py','--model','ehr','--seed',seed,
             '--batch-size','8','--resume')
        step('27_stage11_train_encoders.py','--model','cxr','--seed',seed,
             '--batch-size',args.cxr_batch,'--resume')
    for seed in SEEDS:
        for split in ('train','val'):
            step('28_stage11_cache_features.py','--seed',seed,'--split',split,
                 '--batch-size',args.cxr_batch)
        if seed!=42:
            step('29_stage11_train_alignment.py','--seed',seed,'--smoke')
            step('29_stage11_train_alignment.py','--seed',seed,'--resume')
        step('30_stage11_train_fusion.py','--seed',seed,'--smoke')
        step('30_stage11_train_fusion.py','--seed',seed,'--resume')
    summarize()

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--smoke-only',action='store_true')
    a.add_argument('--cxr-batch',type=int,default=2,help='Use 2 for RTX3050 4GB')
    main(a.parse_args())
