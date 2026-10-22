"""Independent (per-encoder-seed) Stage3-style temporal alignment; no test data."""
import argparse
import json
import math
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from stage11_common import (CACHE,CKPT,LOGS,RESULTS,DEVICE,seed_all,
    atomic_torch,atomic_json,atomic_csv,check_completed,read_feature,check_split_isolation,import_script)

BASE=import_script('08_train_temporal_alignment.py','stage11_alignment_base')

def load(seed):
    a,b=read_feature('train',seed),read_feature('val',seed)
    check_split_isolation(a,b)
    result=[]
    for d in (a,b):
        valid=d['anchor_valid'].reshape(-1).astype(bool)
        if valid.sum()<128:raise RuntimeError(f'Too few valid anchors for seed{seed}')
        result.append((torch.from_numpy(d['ehr_at_cxr'][valid]),
                       torch.from_numpy(d['cxr'][valid])))
    return result

@torch.inference_mode()
def evaluate(model,ev,cv):
    model.eval();e,c=model(ev,cv)
    loss,similarity=BASE.contrastive_loss(e,c,model.logit_scale)
    target=torch.arange(len(ev),device=DEVICE)
    recall=.5*(similarity.argmax(1).eq(target).float().mean()+
               similarity.argmax(0).eq(target).float().mean())
    return dict(val_alignment_loss=float(loss),val_recall_at_1=float(recall))

def run(args):
    if args.seed==42:
        path=CKPT/'trcg_stage3_align_seed42_best.pt'
        if not path.exists():raise FileNotFoundError(path)
        print('REUSE existing Stage3 seed42 alignment:',path,flush=True);return
    if args.seed not in (43,44):raise ValueError('Seed 43 or 44 only')
    seed_all(args.seed)
    (e,c),(ev,cv)=load(args.seed)
    model=BASE.AlignmentProjectors().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(e,c),batch_size=args.batch_size,shuffle=True,num_workers=0)
    ev=ev.to(DEVICE);cv=cv.to(DEVICE)
    tag=f'trcg_stage11_align_seed{args.seed}'
    best=CKPT/(tag+'_best.pt');last=CKPT/(tag+'_last.pt');done=CKPT/(tag+'.done')
    summary=RESULTS/(tag+'_summary.json');hfile=LOGS/(tag+'_history.csv')
    settings=dict(seed=args.seed,epochs=args.epochs,patience=args.patience,
                  lr=args.lr,batch_size=args.batch_size)
    if args.smoke:
        model.train();b1,b2=next(iter(loader));x,y=model(b1.to(DEVICE),b2.to(DEVICE))
        loss,_=BASE.contrastive_loss(x,y,model.logit_scale);loss.backward()
        if not torch.isfinite(loss):raise RuntimeError('Bad alignment smoke loss')
        print(f'SMOKE PASSED alignment seed{args.seed}, loss={float(loss):.4f}',flush=True);return
    if check_completed(done,settings):
        if not best.exists():raise FileNotFoundError(best)
        print('SKIP completed:',tag,flush=True);return
    epoch0=1;bestloss=float('inf');bestep=0;bad=0;history=[]
    if args.resume and last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['settings']!=settings:raise RuntimeError('Resume config mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        epoch0=ck['epoch_next'];bestloss=ck['bestloss'];bestep=ck['bestep']
        bad=ck['bad'];history=ck['history']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        torch.random.set_rng_state(ck['torch_rng'])
        if torch.cuda.is_available() and ck['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        print('RESUME alignment',args.seed,epoch0,flush=True)
    for epoch in range(epoch0,args.epochs+1):
        if bad>=args.patience:break
        model.train();losses=[]
        for a,b in loader:
            a=a.to(DEVICE);b=b.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            x,y=model(a,b);loss,_=BASE.contrastive_loss(x,y,model.logit_scale)
            if not torch.isfinite(loss):raise RuntimeError('Alignment loss NaN')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();losses.append(float(loss.detach()))
        metrics=evaluate(model,ev,cv)
        if metrics['val_alignment_loss']<bestloss-1e-6:
            bestloss=metrics['val_alignment_loss'];bestep=epoch;bad=0
            atomic_torch({'model':model.state_dict(),'epoch':epoch,**metrics,'seed':args.seed},best)
        else:bad+=1
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),**metrics,best_epoch=bestep,bad=bad))
        atomic_csv(history,hfile)
        atomic_torch(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch_next=epoch+1,
             bestloss=bestloss,bestep=bestep,bad=bad,settings=settings,history=history,
             python_rng=random.getstate(),numpy_rng=np.random.get_state(),
             torch_rng=torch.random.get_rng_state(),
             cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),last)
        print(f'align seed{args.seed} epoch {epoch}: val loss={metrics["val_alignment_loss"]:.4f} '
              f'recall@1={metrics["val_recall_at_1"]:.4f}; best={bestep}, bad={bad}/{args.patience}',flush=True)
        if bad>=args.patience:break
    if not best.exists():raise RuntimeError('Missing alignment best')
    saved=torch.load(best,map_location='cpu',weights_only=True)
    result=dict(stage=11,seed=args.seed,best_epoch=saved['epoch'],
        val_alignment_loss=saved['val_alignment_loss'],
        val_recall_at_1=saved['val_recall_at_1'],checkpoint=str(best),settings=settings,
        no_test=True,initialization='Independent seed-specific Stage3 projection')
    atomic_json(result,summary);atomic_json(result,done)
    print('COMPLETE:',tag,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=2e-4)
    run(p.parse_args())
