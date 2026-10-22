"""Matched EHR versus age-gated TRCG fusion for new seeds45,46. VAL only."""
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from stage12_common import (CKPT,LOGS,RESULTS,DEVICE,FIELDS,seed_all,paths,atomic_torch,
     atomic_json,atomic_csv,check_completed,read_feature,check_split_isolation,import_script)

STAGE7=import_script('18_train_stage7_refinement.py','stage11_safe_trcg')

class Cached(Dataset):
    def __init__(self,data):
        self.x={k:torch.as_tensor(data[k],dtype=torch.float32) for k in FIELDS}
        self.y=torch.as_tensor(data['label'],dtype=torch.float32)
        self.ids=data['stay_id'];self.subjects=data['subject_id']
    def __len__(self):return len(self.y)
    def __getitem__(self,i):return {k:v[i] for k,v in self.x.items()},self.y[i],str(self.ids[i])

def make_model(seed):
    E=torch.load(paths('ehr',seed),map_location='cpu',weights_only=True)['model']
    C=torch.load(paths('cxr',seed),map_location='cpu',weights_only=True)['model']
    align_path=(CKPT/'trcg_stage3_align_seed42_best.pt' if seed==42
                else CKPT/f'trcg_stage12_align_seed{seed}_best.pt')
    if not align_path.is_file():raise FileNotFoundError(align_path)
    A=torch.load(align_path,map_location='cpu',weights_only=True)['model']
    esh={'weight':E['head.weight'],'bias':E['head.bias']}
    csh={'weight':C['head.weight'],'bias':C['head.bias']}
    # Fixed gamma is unused in safe_age; learned personalized gate is used.
    return STAGE7.SafeTRCG(esh,csh,A,'safe_age',0.37).to(DEVICE)

@torch.inference_mode()
def evaluate_with_predictions(model,loader):
    model.eval()
    rows=[]
    for x,y,ids in loader:
        x={k:v.to(DEVICE) for k,v in x.items()}
        out=model(x)
        p=out['logits'].float().sigmoid().cpu().numpy()
        baseline=out['ehr_logits'].float().sigmoid().cpu().numpy()
        delta=(out['logits']-out['ehr_logits']).float().cpu().numpy()
        for i,ident in enumerate(ids):
            rows.append(dict(stay_id=str(ident),label=int(y[i]),fused_probability=float(p[i]),
                ehr_probability=float(baseline[i]),cxr_logit_shift=float(delta[i]),
                cxr_age=float(x['cxr_age'][i].item()),
                effective_weight=float(out['effective'][i].item())))
    return rows

def run(args):
    if args.seed not in (45,46):raise ValueError('Stage12 fusion trainer is for seeds 45,46')
    seed_all(args.seed)
    tr,va=read_feature('train',args.seed),read_feature('val',args.seed)
    check_split_isolation(tr,va)
    training=DataLoader(Cached(tr),batch_size=args.batch_size,shuffle=True,num_workers=0)
    valid=DataLoader(Cached(va),batch_size=256,shuffle=False,num_workers=0)
    model=make_model(args.seed)
    # Zero residual must yield identical fused and pretrained EHR predictions.
    model.eval()
    x,y,_=next(iter(valid));x={k:v.to(DEVICE) for k,v in x.items()}
    with torch.no_grad():o=model(x)
    drift=float((o['logits']-o['ehr_logits']).abs().max())
    if drift>1e-5:raise RuntimeError(f'EHR preservation invariant FAILED: {drift}')
    if args.smoke:
        model.train();b,l,_=next(iter(training));b={k:v.to(DEVICE) for k,v in b.items()}
        pos=float(tr['label'].sum());w=torch.tensor((len(tr)-pos)/pos,device=DEVICE)
        loss,_,_=STAGE7.objective(model(b),b,l.to(DEVICE),w,'safe_age')
        loss.backward()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite Stage11 fusion smoke')
        print(f'SMOKE PASSED Stage11 fusion seed{args.seed}: EHR drift={drift:.1e}, loss={float(loss):.4f}',flush=True)
        return
    tag=f'trcg_stage12_safe_age_seed{args.seed}'
    best=CKPT/(tag+'_best.pt');last=CKPT/(tag+'_last.pt');done=CKPT/(tag+'.done')
    history_path=LOGS/(tag+'_history.csv');summary_path=RESULTS/(tag+'_summary.json')
    pred_path=RESULTS/(tag+'_val_predictions.csv')
    settings=dict(seed=args.seed,variant='safe_age',lr=args.lr,batch_size=args.batch_size,
                  patience=args.patience,epochs=args.epochs,lambda_comp=0.02,
                  age_regularizer=True,zero_residual=True,frozen_ehr_head=True,
                  initialized_from_seed_specific_alignment=True)
    if check_completed(done,settings):
        for p in (best,pred_path,summary_path):
            if not p.exists():raise RuntimeError(f'Completed but missing {p}')
        print('SKIP already finished',tag,flush=True);return
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-4)
    pos=float(tr['label'].sum());w=torch.tensor((len(tr)-pos)/pos,device=DEVICE)
    start=1;best_ap=-1.;best_auc=-1.;best_ep=0;bad=0;history=[]
    if args.resume and last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state['settings']!=settings:raise RuntimeError('Resume settings changed')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        start=state['epoch_next'];best_ap=state['best_ap'];best_auc=state['best_auc'];
        best_ep=state['best_ep'];bad=state['bad'];history=state['history']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        torch.random.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available() and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        print('RESUME',tag,'epoch',start,flush=True)
    for ep in range(start,args.epochs+1):
        if bad>=args.patience:break
        model.train();trainloss=[]
        for batch,labels,_ in training:
            batch={k:v.to(DEVICE) for k,v in batch.items()};labels=labels.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            output=model(batch)
            loss,_,_=STAGE7.objective(output,batch,labels,w,'safe_age')
            if not torch.isfinite(loss):raise RuntimeError('Bad loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0)
            opt.step();trainloss.append(float(loss.detach()))
        metrics=STAGE7.evaluate(model,valid)
        better=metrics['val_auprc']>best_ap+1e-6 or (
            abs(metrics['val_auprc']-best_ap)<1e-6 and metrics['val_auroc']>best_auc+1e-6)
        if better:
            best_ap=metrics['val_auprc'];best_auc=metrics['val_auroc'];best_ep=ep;bad=0
            atomic_torch({'model':model.state_dict(),'epoch':ep,'metrics':metrics,'settings':settings},best)
        else:bad+=1
        history.append(dict(epoch=ep,train_loss=float(np.mean(trainloss)),
             val_auprc=metrics['val_auprc'],val_auroc=metrics['val_auroc'],
             ehr_anchor_auprc=metrics['ehr_base_auprc'],best_epoch=best_ep,bad=bad))
        atomic_csv(history,history_path)
        atomic_torch(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch_next=ep+1,
            best_ap=best_ap,best_auc=best_auc,best_ep=best_ep,bad=bad,
            history=history,settings=settings,python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),torch_rng=torch.random.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),last)
        print(f'{tag} epoch {ep}: AUPRC={metrics["val_auprc"]:.4f}, '
            f'EHR={metrics["ehr_base_auprc"]:.4f}, AUROC={metrics["val_auroc"]:.4f} '
            f'best={best_ep}, bad={bad}/{args.patience}',flush=True)
        if bad>=args.patience:break
    if not best.exists():raise RuntimeError('No best fusion checkpoint')
    saved=torch.load(best,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model'],strict=True)
    predictions=evaluate_with_predictions(model,valid)
    by_subject=dict(zip(va['stay_id'],va['subject_id']))
    for item in predictions:item['subject_id']=str(by_subject[item['stay_id']])
    atomic_csv(predictions,pred_path)
    report=dict(stage=11,seed=args.seed,variant='safe_age',best_epoch=int(saved['epoch']),
        val_auprc=saved['metrics']['val_auprc'],val_auroc=saved['metrics']['val_auroc'],
        ehr_anchor_auprc=saved['metrics']['ehr_base_auprc'],
        val_ap_gain_vs_matched_ehr=float(saved['metrics']['val_auprc']-saved['metrics']['ehr_base_auprc']),
        checkpoint=str(best),validation_predictions=str(pred_path),settings=settings,
        test_accessed=False,encoder_weights_independently_trained_seed=args.seed)
    atomic_json(report,summary_path);atomic_json(report,done)
    print('COMPLETE:',tag,'val AUPRC',round(report['val_auprc'],5),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--epochs',type=int,default=16);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=3e-5)
    run(p.parse_args())
