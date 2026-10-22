"""Validation-locked 5-seed ensemble and separate retrospective test evaluation.
Usage:
 python scripts/36_stage12_finalize.py --val-only
 python scripts/36_stage12_finalize.py --final-test
Never uses test labels or predictions to select model or fit temperatures.
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from stage12_common import (ROOT,CACHE,CKPT,RESULTS,DEVICE,ALL_SEEDS,FIELDS,
    paths,align_path,fusion_path,read_feature,atomic_json,atomic_csv,import_script)

STAGE7=import_script('18_train_stage7_refinement.py','stage12_safe_trcg_ref')


def build_model(seed):
    ehr=torch.load(paths('ehr',seed),map_location='cpu',weights_only=True)['model']
    cxr=torch.load(paths('cxr',seed),map_location='cpu',weights_only=True)['model']
    align=torch.load(align_path(seed),map_location='cpu',weights_only=True)['model']
    ehr_head={'weight':ehr['head.weight'],'bias':ehr['head.bias']}
    cxr_head={'weight':cxr['head.weight'],'bias':cxr['head.bias']}
    model=STAGE7.SafeTRCG(ehr_head,cxr_head,align,'safe_age',0.37)
    ck=torch.load(fusion_path(seed),map_location='cpu',weights_only=True)
    model.load_state_dict(ck['model'],strict=True)
    return model.to(DEVICE).eval()


@torch.inference_mode()
def predict(seed,split,batch_size=256):
    data=read_feature(split,seed)
    model=build_model(seed)
    e=[];f=[]
    for start in range(0,len(data['label']),batch_size):
        end=min(len(data['label']),start+batch_size)
        x={key:torch.from_numpy(data[key][start:end]).to(DEVICE) for key in FIELDS}
        out=model(x)
        e.append(out['ehr_logits'].float().sigmoid().cpu().numpy().reshape(-1))
        f.append(out['logits'].float().sigmoid().cpu().numpy().reshape(-1))
    del model
    return pd.DataFrame(dict(stay_id=data['stay_id'],subject_id=data['subject_id'],
        label=data['label'],**{f'ehr_{seed}':np.concatenate(e),f'fusion_{seed}':np.concatenate(f)}))


def prediction_frame(split):
    tables=[]
    for seed in ALL_SEEDS:
        t=predict(seed,split)
        if len(t)!=len(t.stay_id.unique()):raise RuntimeError(f'Duplicate stay in seed {seed}')
        if split=='val':
            name=(f'trcg_stage11_safe_age_seed{seed}_val_predictions.csv' if seed<=44
                  else f'trcg_stage12_safe_age_seed{seed}_val_predictions.csv')
            old=RESULTS/name
            if not old.exists():raise FileNotFoundError(old)
            prev=pd.read_csv(old,dtype={'stay_id':str})
            if set(prev.stay_id)!=set(t.stay_id):raise RuntimeError(f'Val ID mismatch seed{seed}')
            chk=prev.set_index('stay_id').loc[t.stay_id]
            for reference,ours in (('fused_probability',f'fusion_{seed}'),('ehr_probability',f'ehr_{seed}')):
                drift=np.max(abs(chk[reference].to_numpy()-t[ours].to_numpy()))
                if drift>1e-4: raise RuntimeError(f'VAL checkpoint prediction mismatch seed{seed}, {ours}: {drift}')
        table=t.set_index('stay_id')
        if tables:
            # All seeds must have identical patient IDs / labels, not just matching lengths.
            ref=tables[0]
            if set(table.index)!=set(ref.index):raise RuntimeError(f'{split} seed{seed}: cohort differs')
            table=table.loc[ref.index]
            if not np.array_equal(table['label'],ref['label']):raise RuntimeError('Labels differ across seeds')
            if not np.array_equal(table['subject_id'],ref['subject_id']):raise RuntimeError('Subjects differ across seeds')
            table=table[[f'ehr_{seed}',f'fusion_{seed}']]
        tables.append(table)
    result=pd.concat(tables,axis=1).reset_index()
    result['ehr_mean']=result[[f'ehr_{i}' for i in ALL_SEEDS]].mean(axis=1)
    result['fusion_mean']=result[[f'fusion_{i}' for i in ALL_SEEDS]].mean(axis=1)
    if not result.select_dtypes(include=['number']).apply(np.isfinite).all().all():
        raise RuntimeError('Nonfinite predictions')
    return result


def metrics(y,p):
    return dict(auprc=float(average_precision_score(y,p)),auroc=float(roc_auc_score(y,p)),
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])))


def fit_temperature(y,p):
    if set(np.unique(y))!={0,1}:raise RuntimeError('Both classes needed for temperature fitting')
    logits=logit(np.clip(p,1e-6,1-1e-6))
    def loss(t):
        pred=expit(logits/t)
        return log_loss(y,np.clip(pred,1e-7,1-1e-7),labels=[0,1])
    result=minimize_scalar(loss,bounds=(0.5,10.),method='bounded',options={'xatol':1e-5})
    if not result.success:raise RuntimeError('Temperature fitting failed')
    return float(result.x)


def apply_temperature(p,t):
    return expit(logit(np.clip(p,1e-6,1-1e-6))/t)


def cluster_bootstrap(df,predA,predB,replicates=1500):
    ids=df['subject_id'].astype(str).to_numpy()
    y=df['label'].to_numpy(int)
    subs=np.unique(ids)
    locations={s:np.flatnonzero(ids==s) for s in subs}
    rng=np.random.default_rng(12046)
    diffs=[];da=[];db=[]
    a=df[predA].to_numpy(float);b=df[predB].to_numpy(float)
    for _ in range(replicates):
        chosen=rng.choice(subs,size=len(subs),replace=True)
        index=np.concatenate([locations[s] for s in chosen])
        labels=y[index]
        if len(np.unique(labels))<2:continue
        diffs.append(average_precision_score(labels,a[index])-average_precision_score(labels,b[index]))
        da.append(roc_auc_score(labels,a[index])-roc_auc_score(labels,b[index]))
        db.append(brier_score_loss(labels,a[index])-brier_score_loss(labels,b[index]))
    if len(diffs)<int(replicates*.95):raise RuntimeError('Insufficient valid cluster resamples')
    return dict(method='subject_cluster_bootstrap',replicates=len(diffs),seed=12046,
        fusion_minus_ehr_auprc_95ci=np.quantile(diffs,[.025,.975]).tolist(),
        fusion_minus_ehr_auroc_95ci=np.quantile(da,[.025,.975]).tolist(),
        fusion_minus_ehr_brier_95ci=np.quantile(db,[.025,.975]).tolist())


def do_validation():
    df=prediction_frame('val')
    if len(df)!=490:raise RuntimeError(f'Expected 490 validation stays, found {len(df)}')
    y=df.label.to_numpy(int)
    temps={};report={}
    for name in ('ehr','fusion'):
        raw=df[f'{name}_mean'].to_numpy(float)
        t=fit_temperature(y,raw)
        temps[name]=t
        df[f'{name}_calibrated']=apply_temperature(raw,t)
        report[name]=dict(temperature=t,uncalibrated=metrics(y,raw),calibrated=metrics(y,df[f'{name}_calibrated']))
    report['delta_fusion_minus_ehr_auprc_uncalibrated']=report['fusion']['uncalibrated']['auprc']-report['ehr']['uncalibrated']['auprc']
    summary={'stage':12,'scope':'validation_only','predefined_configuration':'safe_age; matched EHR anchor; five independent seeds 42-46',
        'seeds':list(ALL_SEEDS),'validation_n':len(df),'validation_deaths':int(y.sum()),
        'validation':report,'temperature_fit':'unweighted validation NLL after equal-weight probability ensembling',
        'limitations':['Repeatedly inspected validation cohort; no fresh hyperparameter search.',
                       'Five seeds share same ImageNet starting checkpoint and fixed patient split.',
                       'Test set evaluated separately on explicit request.']}
    atomic_csv(df.to_dict('records'),RESULTS/'trcg_stage12_five_seed_validation_predictions.csv')
    atomic_json(summary,RESULTS/'trcg_stage12_validation_summary.json')
    atomic_json(dict(temperatures=temps,stage=12,seeds=list(ALL_SEEDS),
                     source='validation_only',fitting='unweighted NLL'),
                RESULTS/'trcg_stage12_locked_temperatures.json')
    print('Stage 12 VAL ONLY: EHR AP',round(report['ehr']['uncalibrated']['auprc'],5),
          'Fusion AP',round(report['fusion']['uncalibrated']['auprc'],5),
          'Temperatures',temps,flush=True)
    return summary


def do_final_test():
    # Load preexisting locked validation temperatures, do not tune using test.
    temperatures_path=RESULTS/'trcg_stage12_locked_temperatures.json'
    validation_path=RESULTS/'trcg_stage12_validation_summary.json'
    if not temperatures_path.exists() or not validation_path.exists():
        raise RuntimeError('Run --val-only first to lock temperatures and validation output')
    locked=json.loads(temperatures_path.read_text(encoding='utf8'))
    if locked['seeds']!=list(ALL_SEEDS) or locked['source']!='validation_only':
        raise RuntimeError('Invalid temperature provenance')
    out=RESULTS/'trcg_stage12_final_test_summary.json'
    if out.exists():
        print('Final test report already exists; leaving it unchanged:',out,flush=True)
        return
    df=prediction_frame('test')
    if len(df)!=1231:raise RuntimeError(f'Expected 1231 test stays, found {len(df)}')
    y=df.label.to_numpy(int)
    metrics_out={};per_seed=[]
    for seed in ALL_SEEDS:
        for name in ('ehr','fusion'):
            per_seed.append(dict(seed=seed,model=name,**metrics(y,df[f'{name}_{seed}'])))
    for name in ('ehr','fusion'):
        p=df[f'{name}_mean'].to_numpy(float)
        t=float(locked['temperatures'][name])
        df[f'{name}_calibrated']=apply_temperature(p,t)
        metrics_out[name]=dict(temperature=t,uncalibrated=metrics(y,p),
                               calibrated=metrics(y,df[f'{name}_calibrated']))
    bootstrap=cluster_bootstrap(df,'fusion_calibrated','ehr_calibrated')
    summary={'stage':12,'scope':'retrospective_benchmark_test','n':len(df),
             'deaths':int(y.sum()),'seeds':list(ALL_SEEDS),'ensemble':metrics_out,
             'per_seed':per_seed,'paired_cluster_bootstrap':bootstrap,
             'calibration_source':'validation only; settings frozen before test inference',
             'selected_architecture':'safe_age fusion and matched EHR-only; no test selection',
             'caveats':['The benchmark test was inspected in earlier project stages; not a fresh blind test.',
                        'Bootstraps reflect patient resampling uncertainty, not model-selection or dataset-shift uncertainty.',
                        'Five independent training seeds share ImageNet initialization and the same split.']}
    atomic_csv(per_seed,RESULTS/'trcg_stage12_per_seed_test.csv')
    atomic_csv(df.to_dict('records'),RESULTS/'trcg_stage12_five_seed_test_predictions.csv')
    atomic_json(summary,out)
    print('Stage 12 TEST: EHR AP',round(metrics_out['ehr']['calibrated']['auprc'],5),
          'Fusion AP',round(metrics_out['fusion']['calibrated']['auprc'],5),flush=True)
    print('Saved',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    options=p.add_mutually_exclusive_group(required=True)
    options.add_argument('--val-only',action='store_true')
    options.add_argument('--final-test',action='store_true')
    a=p.parse_args()
    if a.val_only:do_validation()
    else:do_final_test()
