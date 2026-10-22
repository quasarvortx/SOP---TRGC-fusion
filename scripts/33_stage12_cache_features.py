"""Cache seed-matched representations; train/val for seeds45,46, test for all five seeds."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from stage12_common import (CACHE,CKPT,RESULTS,DEVICE,AMP,seed_all,paths,
    atomic_npz,atomic_json,import_script,check_split_isolation,feature_path)
from src.models.trcg_encoders import EHRMortality,CXRMortality

STAGE3=import_script('08_train_temporal_alignment.py','stage11_orig_stage3')

def prepare_manifest():
    manifest=pd.read_csv(CACHE/'trcg_manifest.csv',dtype={'stay_id':str,'subject_id':str})
    if manifest.stay_id.duplicated().any():raise RuntimeError('Duplicate stay IDs in manifest')
    t=manifest[manifest.split=='train'];v=manifest[manifest.split=='val']
    check_split_isolation({'stay_id':t.stay_id.astype(str),'subject_id':t.subject_id.astype(str)},
                          {'stay_id':v.stay_id.astype(str),'subject_id':v.subject_id.astype(str)})
    return manifest

def copy_42(split,manifest,out):
    old=CACHE/f'trcg_stage4_{split}_features.npz'
    if not old.exists():raise FileNotFoundError(old)
    with np.load(old,allow_pickle=False) as z: data={k:np.asarray(z[k]).copy() for k in z.files}
    frame=manifest[manifest.split==split].set_index('stay_id')
    ids=data['stay_id'].astype(str)
    if set(ids)!=set(frame.index):raise RuntimeError(f'Seed42 cache IDs differ from manifest: {split}')
    subj=np.asarray(frame.loc[ids,'subject_id'].astype(str).to_numpy(),dtype=str)
    if not np.array_equal(data['label'].astype(int),frame.loc[ids,'label'].to_numpy().astype(int)):
        raise RuntimeError('Seed42 cache labels differ from manifest')
    data['subject_id']=subj
    data['source_seed']=np.array([42],dtype=np.int64)
    atomic_npz(out,**data)
    print('COPY verified original seed42 frozen features:',out,flush=True)

def load_models(seed):
    prep=json.loads((CACHE/'ehr_preprocessor.json').read_text(encoding='utf8'))
    ehr=EHRMortality(prep['categorical_vocab'])
    cxr=CXRMortality(None)
    for kind,mod in (('ehr',ehr),('cxr',cxr)):
        p=paths(kind,seed)
        if not p.exists():raise FileNotFoundError(p)
        ck=torch.load(p,map_location='cpu',weights_only=True)
        mod.load_state_dict(ck['model'],strict=True)
        mod.requires_grad_(False)
    return ehr.to(DEVICE).eval(),cxr.to(DEVICE).eval()

@torch.inference_mode()
def extract(split,seed,batch_size,manifest,out):
    ds=STAGE3.PairedStage3(split)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0,
        pin_memory=AMP,collate_fn=STAGE3.collate)
    ehrmodel,cxrmodel=load_models(seed)
    # Immutable EHR quality descriptors. Must join via stay_id rather than row position.
    quality_path=CACHE/f'trcg_stage4_{split}_features.npz'
    with np.load(quality_path,allow_pickle=False) as z:
        source_ids=z['stay_id'].astype(str)
        if len(source_ids)!=len(set(source_ids)):raise RuntimeError('Duplicate cached quality IDs')
        quality_by_id={key:np.asarray(z['ehr_quality'][i],dtype='float32') for i,key in enumerate(source_ids)}
    output={key:[] for key in ('ehr_at_48','ehr_at_cxr','cxr','cxr_age','cxr_hour',
        'anchor_valid','ehr_quality','cxr_entropy','label','stay_id','subject_id')}
    df=manifest[manifest.split==split].set_index('stay_id')
    cxr_y=[];cxr_p=[]
    for ehr,img,hours,labels,ids in tqdm(loader,desc=f'Seed {seed} {split} feature cache'):
        ehr={k:v.to(DEVICE,non_blocking=True) for k,v in ehr.items()}
        img=img.to(DEVICE,non_blocking=True);hours=hours.to(DEVICE)
        with torch.amp.autocast('cuda',enabled=AMP):
            final,seq=ehrmodel.encoder(ehr,return_sequence=True)
            visual=cxrmodel.encoder(img)
        final=final.float();seq=seq.float();visual=visual.float()
        time_mask=torch.arange(seq.shape[1],device=DEVICE)[None,:]<ehr['lengths'][:,None]
        past=time_mask & (ehr['times']<=hours[:,None]+1e-6)
        index=past.long().sum(1)-1
        is_valid=index>=0
        anchor=seq[torch.arange(len(ids),device=DEVICE),index.clamp(min=0)]
        anchor[~is_valid]=0.
        cxr_logit=cxrmodel.head(visual).float().squeeze(-1)
        cxr_probability=torch.sigmoid(cxr_logit)
        eps=1e-7
        entropy=-(cxr_probability.clamp(eps,1-eps)*torch.log(cxr_probability.clamp(eps,1-eps))+
                  (1-cxr_probability).clamp(eps,1-eps)*torch.log((1-cxr_probability).clamp(eps,1-eps)))
        if any(i not in df.index for i in ids):raise RuntimeError('Unknown stay in cache')
        subject=np.asarray(df.loc[ids,'subject_id'].astype(str).to_numpy(),dtype=str)
        manifest_labels=df.loc[ids,'label'].to_numpy(dtype=np.int64)
        if not np.array_equal(manifest_labels,labels):raise RuntimeError('Label mismatch during caching')
        d=dict(ehr_at_48=final.cpu().numpy(),ehr_at_cxr=anchor.cpu().numpy(),
               cxr=visual.cpu().numpy(),cxr_age=(48.-hours).cpu().numpy(),cxr_hour=hours.cpu().numpy(),
               anchor_valid=is_valid.cpu().numpy().astype(np.float32),
               ehr_quality=np.stack([quality_by_id[str(i)] for i in ids]),
               cxr_entropy=entropy.cpu().numpy(),label=np.asarray(labels,dtype=np.int64),
               stay_id=np.array(ids,dtype=str),subject_id=subject)
        for k,a in d.items():output[k].append(a)
        cxr_y.extend(labels);cxr_p.extend(cxr_probability.cpu().tolist())
    data={k:np.concatenate(v) for k,v in output.items()}
    if set(data['stay_id'].astype(str))!=set(df.index):raise RuntimeError('Missing cached stay(s)')
    if not np.isfinite(data['ehr_at_48']).all() or not np.isfinite(data['cxr']).all():
        raise RuntimeError('Nonfinite encoder features')
    data['source_seed']=np.array([seed],dtype=np.int64)
    atomic_npz(out,**data)
    metric=float(average_precision_score(cxr_y,cxr_p))
    print(f'Saved {len(data["stay_id"])} pairs, {int(data["anchor_valid"].sum())} valid anchors; '
          f'CXR-only {split} AP={metric:.4f}: {out}',flush=True)
    return metric

def run(args):
    if args.seed not in (42,43,44,45,46):raise ValueError('Seed must be 42 through 46')
    seed_all(args.seed)
    manifest=prepare_manifest()
    out=feature_path(args.split,args.seed)
    if out.exists() and not args.force:
        with np.load(out,allow_pickle=False) as z:
            if int(z['source_seed'][0])!=args.seed:raise RuntimeError('Cache seed mismatch')
            ids=z['stay_id'].astype(str)
        expected=set(manifest.loc[manifest.split==args.split,'stay_id'])
        if len(set(ids))!=len(ids) or set(ids)!=expected:
            raise RuntimeError('Existing cache does not match manifest. Investigate before deleting/forcing.')
        print('SKIP verified cache',out,flush=True);return
    if args.seed==42 and (CACHE/f'trcg_stage4_{args.split}_features.npz').exists():
        copy_42(args.split,manifest,out)
    elif args.seed in (43,44) and args.split!='test':
        raise RuntimeError('Use existing Stage11 cache for these seeds; do not duplicate')
    else:extract(args.split,args.seed,args.batch_size,manifest,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--split',choices=['train','val','test'],required=True)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--force',action='store_true',help='Overwrite existing Stage11 cache ONLY, not earlier stages')
    run(p.parse_args())
