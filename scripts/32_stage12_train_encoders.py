"""Stage12: train new independent seeds 45,46 on TRAIN, select using VAL; never read TEST.
Examples:
 python scripts/32_stage12_train_encoders.py --model ehr --seed 45 --smoke
 python scripts/32_stage12_train_encoders.py --model cxr --seed 45 --resume
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from stage12_common import (ROOT,CACHE,CKPT,LOGS,RESULTS,DEVICE,AMP,seed_all,
    atomic_torch,atomic_json,atomic_csv,check_completed,import_script)

BASE=import_script('06_train_unimodal.py','stage11_base_unimodal')

def create(args):
    if args.seed not in (45,46):raise ValueError('This trainer is for seeds 45 and 46 only.')
    if args.model=='ehr':
        tr=BASE.EHREventDataset('train');va=BASE.EHREventDataset('val');collate=BASE.ehr_collate
    else:
        tr=BASE.CXRDataset('train');va=BASE.CXRDataset('val');collate=None
    # Original Stage 2 architecture and ImageNet initialization (for CXR)
    model=BASE.make_model(args.model).to(DEVICE)
    loaders=[DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,collate_fn=collate,
                        num_workers=0,pin_memory=AMP) for ds,shuffle in ((tr,True),(va,False))]
    labels=([int(s['label']) for s in tr.items] if args.model=='ehr'
            else tr.rows['label'].astype(int).tolist())
    if len(labels)==0 or min(sum(labels),len(labels)-sum(labels))<=0:raise RuntimeError('Bad class balance')
    weight=torch.tensor((len(labels)-sum(labels))/sum(labels),device=DEVICE)
    return model,loaders,weight

@torch.inference_mode()
def evaluate(model,loader,loss_fn):
    model.eval(); yall=[];pall=[]; total_loss=0.; n=0
    for x,y,_ in loader:
        x=BASE.to_device(x);y=y.to(DEVICE)
        with torch.amp.autocast('cuda',enabled=AMP): pred=model(x)
        loss=loss_fn(pred.float(),y.float())
        yall.extend(y.detach().cpu().tolist());pall.extend(pred.float().sigmoid().cpu().tolist())
        total_loss+=float(loss)*len(y);n+=len(y)
    return dict(val_loss=total_loss/n,val_auroc=float(roc_auc_score(yall,pall)),
                val_auprc=float(average_precision_score(yall,pall)))

def run(args):
    if not AMP and not args.smoke:raise RuntimeError('CUDA required for the full Stage11 run')
    seed_all(args.seed)
    model,(train,val),weight=create(args)
    lr=args.lr if args.lr is not None else (3e-4 if args.model=='ehr' else 2e-5)
    optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=AMP)
    crit=nn.BCEWithLogitsLoss(pos_weight=weight)
    settings=dict(kind=args.model,seed=args.seed,epochs=args.epochs,patience=args.patience,
                  lr=lr,batch_size=args.batch_size,architecture='stage2_exact')
    prefix=f'trcg_stage12_{args.model}_seed{args.seed}'
    best=CKPT/(prefix+'_best.pt');last=CKPT/(prefix+'_last.pt');done=CKPT/(prefix+'.done')
    history_path=LOGS/(prefix+'_history.csv');summary_path=RESULTS/(prefix+'_summary.json')
    if args.smoke:
        model.train(); x,y,_=next(iter(train));x=BASE.to_device(x);y=y.to(DEVICE)
        with torch.amp.autocast('cuda',enabled=AMP): out=model(x)
        loss=crit(out.float(),y.float());scaler.scale(loss).backward()
        if not math.isfinite(float(loss)) or model.head.weight.grad is None:
            raise RuntimeError('Encoder smoke backward failed')
        print(f'SMOKE PASSED {args.model} seed{args.seed}: loss={float(loss):.4f}',flush=True)
        return
    if check_completed(done,settings):
        if not best.exists():raise RuntimeError(f'Missing best checkpoint for completed {prefix}')
        print('SKIP already complete:',prefix,flush=True);return
    ep0=1;bad=0;best_ap=-1.;best_auc=-1.;best_ep=0;history=[]
    if args.resume and last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state['settings']!=settings:raise RuntimeError(f'Resume settings changed: {last}')
        model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        ep0=state['epoch_next'];bad=state['bad'];best_ap=state['best_ap']
        best_auc=state['best_auc'];best_ep=state['best_ep'];history=state['history']
        torch.random.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng'])
        import random
        random.setstate(state['python_rng'])
        if AMP and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        print('RESUME',prefix,'at epoch',ep0,flush=True)
    for ep in range(ep0,args.epochs+1):
        if bad>=args.patience:break
        model.train(); losses=[]
        for x,y,_ in tqdm(train,desc=f'{prefix} epoch {ep}',leave=False):
            x=BASE.to_device(x);y=y.to(DEVICE)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=AMP):out=model(x)
            loss=crit(out.float(),y.float())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optim);scaler.update();losses.append(float(loss))
        metrics=evaluate(model,val,crit)
        better=(metrics['val_auprc']>best_ap+1e-6 or
                (abs(metrics['val_auprc']-best_ap)<1e-6 and metrics['val_auroc']>best_auc+1e-6))
        if better:
            best_ap=metrics['val_auprc'];best_auc=metrics['val_auroc'];best_ep=ep;bad=0
            atomic_torch(dict(model=model.state_dict(),epoch=ep,seed=args.seed,
                model_name=args.model,**metrics),best)
        else:bad+=1
        history.append(dict(epoch=ep,train_loss=float(np.mean(losses)),**metrics,best_epoch=best_ep,bad=bad))
        import random
        atomic_torch(dict(model=model.state_dict(),optimizer=optim.state_dict(),
            scaler=scaler.state_dict(),epoch_next=ep+1,bad=bad,best_ap=best_ap,
            best_auc=best_auc,best_ep=best_ep,history=history,settings=settings,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.random.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if AMP else None),last)
        atomic_csv(history,history_path)
        print(f'{prefix} epoch {ep}: val AUROC={metrics["val_auroc"]:.4f} '
            f'AUPRC={metrics["val_auprc"]:.4f} best={best_ep} patience={bad}/{args.patience}',flush=True)
        if bad>=args.patience:break
    if not best.exists():raise RuntimeError(f'Best checkpoint missing: {best}')
    ck=torch.load(best,map_location='cpu',weights_only=True)
    result=dict(stage=11,kind=args.model,seed=args.seed,best_epoch=int(ck['epoch']),
                val_auprc=ck['val_auprc'],val_auroc=ck['val_auroc'],settings=settings,
                checkpoint=str(best),scope='validation_only',test_accessed=False)
    atomic_json(result,summary_path);atomic_json(result,done)
    print('COMPLETE:',prefix,'best val AUPRC',round(best_ap,4),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['ehr','cxr'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--patience',type=int,default=8);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=None)
    run(p.parse_args())
