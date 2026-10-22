"""TRCG Stage 4B: cautious EHR then CXR encoder finetuning.
Requires original src/models/trcg_encoders.py, src/models/trcg_fusion.py,
scripts/08_train_temporal_alignment.py, Stage 2 checkpoints and Stage 4A caches.
"""
import argparse, csv, importlib.util, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.trcg_fusion import TRCGFusion
from src.models.trcg_encoders import EHRMortality, CXRMortality

spec=importlib.util.spec_from_file_location('trcg_stage3_loader', ROOT/'scripts'/'08_train_temporal_alignment.py')
stage3=importlib.util.module_from_spec(spec); spec.loader.exec_module(stage3)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP=DEVICE.type=='cuda'
CKPT=ROOT/'checkpoints'; CACHE=ROOT/'cache'; RESULTS=ROOT/'results'; LOGS=ROOT/'logs'
for p in (CKPT,RESULTS,LOGS): p.mkdir(exist_ok=True)

class PairedWithQuality(stage3.PairedStage3):
    def __init__(self, split):
        super().__init__(split)
        with np.load(CACHE/f'trcg_stage4_{split}_features.npz') as data:
            self.q_by_id={str(k): (data['ehr_quality'][i].copy(),float(data['cxr_entropy'][i]),float(data['anchor_valid'][i])) for i,k in enumerate(data['stay_id'])}
        missing=[str(r['stay_id']) for r in self.rows if str(r['stay_id']) not in self.q_by_id]
        if missing: raise RuntimeError('Missing quality rows: '+str(missing[:5]))
    def __getitem__(self,i):
        sample=super().__getitem__(i)
        key=sample[-1]
        return (*sample,self.q_by_id[key])

def collate(samples):
    base=[row[:5] for row in samples]
    ehr,img,hr,lab,ids=stage3.collate(base)
    extras=[row[5] for row in samples]
    q=torch.tensor(np.stack([e[0] for e in extras]),dtype=torch.float32)
    ent=torch.tensor([e[1] for e in extras],dtype=torch.float32)
    valid=torch.tensor([e[2] for e in extras],dtype=torch.float32)
    return ehr,img,hr,torch.tensor(lab,dtype=torch.float32),ids,q,ent,valid

class FullTRCG(nn.Module):
    def __init__(self):
        super().__init__()
        prep=json.loads((CACHE/'ehr_preprocessor.json').read_text(encoding='utf8'))
        self.ehr=EHRMortality(prep['categorical_vocab'])
        self.cxr=CXRMortality(None)
        self.ehr.load_state_dict(torch.load(CKPT/'trcg_stage2_ehr_seed42_best.pt',map_location='cpu',weights_only=True)['model'],strict=True)
        self.cxr.load_state_dict(torch.load(CKPT/'trcg_stage2_cxr_seed42_best.pt',map_location='cpu',weights_only=True)['model'],strict=True)
        es={'weight':self.ehr.head.weight.detach().clone(),'bias':self.ehr.head.bias.detach().clone()}
        cs={'weight':self.cxr.head.weight.detach().clone(),'bias':self.cxr.head.bias.detach().clone()}
        self.fusion=TRCGFusion(es,cs)
        frozen=torch.load(CKPT/'trcg_stage4a_v2_seed42_best.pt',map_location='cpu',weights_only=True)
        self.fusion.load_state_dict(frozen['model'],strict=True)

    def forward(self,ehr,img,cxr_hr,q,ent):
        # Strict causal temporal anchor: last real EHR timestamp <= CXR time.
        zfinal,seq=self.ehr.encoder(ehr,return_sequence=True)
        B,T,_=seq.shape
        valid_pos=torch.arange(T,device=seq.device)[None,:] < ehr['lengths'][:,None]
        before=valid_pos & (ehr['times']<=cxr_hr[:,None]+1e-6)
        idx=before.long().sum(1)-1
        valid=(idx>=0).float()
        anchor=seq[torch.arange(B,device=seq.device),idx.clamp(min=0)]
        anchor=anchor*valid[:,None]
        # CXR backbone. Phase B uses no_grad via requires_grad flags + eval mode.
        zcxr=self.cxr.encoder(img)
        b={'ehr_at_48':zfinal.float(),'ehr_at_cxr':anchor.float(),
           'cxr':zcxr.float(),'cxr_age':(48-cxr_hr).float(),
           'anchor_valid':valid,'ehr_quality':q.float(),'cxr_entropy':ent.float()}
        return self.fusion(b),b

def mode(model,phase):
    for p in model.parameters():p.requires_grad_(False)
    # Stage 4A head is already trained: unfreeze only fusion layers with small LR.
    for p in model.fusion.parameters():p.requires_grad_(True)
    # EHR last transformer block + norm (Phase B and C).
    for p in model.ehr.encoder.encoder.layers[-1].parameters():p.requires_grad_(True)
    for p in model.ehr.encoder.out_norm.parameters():p.requires_grad_(True)
    if phase=='C':
        # Only final ResNet block + head normalization parameters are updated.
        for p in model.cxr.encoder.layer4.parameters():p.requires_grad_(True)
        for p in model.cxr.encoder.fc.parameters():p.requires_grad_(True)
    model.train()
    # Frozen batch-norm layers should not update running statistics.
    if phase=='B':model.cxr.eval()
    else:
        model.cxr.eval()
        model.cxr.encoder.layer4.train()
    # Frozen earlier EHR layers have no BatchNorm.

def loss_fn(out,b,y,pos_weight,align_weight=0.005):
    bce=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mort=bce(out['logits'].float(),y)
    uni=bce(out['ehr_logits'].float(),y)+0.25*bce(out['cxr_logits'].float(),y)
    v=b['anchor_valid'].bool()
    alignment=mort.new_zeros(())
    if align_weight and v.sum()>=2:
        a=out['q_ehr'][v].float(); c=out['q_cxr'][v].float()
        sim=(a@c.T)/0.07
        labels=torch.arange(len(a),device=sim.device)
        alignment=(nn.functional.cross_entropy(sim,labels)+nn.functional.cross_entropy(sim.T,labels))/2
    return mort+0.1*uni+align_weight*alignment

def batches(loader,model,crit,phase,optimizer=None,scaler=None,accum=4,smoke=False):
    training=optimizer is not None
    model.train(training)
    if training: mode(model,phase);optimizer.zero_grad(set_to_none=True)
    ys=[];ps=[];ids_out=[]; losses=[]
    for k,(ehr,img,hr,y,ids,q,ent,_) in enumerate(tqdm(loader,desc=('Train' if training else 'Val'),leave=False)):
        ehr={n:v.to(DEVICE,non_blocking=True) for n,v in ehr.items()}
        img=img.to(DEVICE,non_blocking=True);hr=hr.to(DEVICE)
        y=y.to(DEVICE);q=q.to(DEVICE);ent=ent.to(DEVICE)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast('cuda',enabled=AMP):
                out,b=model(ehr,img,hr,q,ent)
            loss=loss_fn(out,b,y,crit)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            if training:
                scaler.scale(loss/accum).backward()
                if ((k+1)%accum==0) or (k+1==len(loader)):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        ys.extend(y.detach().cpu().tolist());ps.extend(out['logits'].detach().float().sigmoid().cpu().tolist());ids_out.extend(ids)
        if smoke:break
    return {'loss':float(np.mean(losses)),'auroc':float(roc_auc_score(ys,ps)) if not smoke else 0.,
            'auprc':float(average_precision_score(ys,ps)) if not smoke else 0.,'ids':ids_out,'y':ys,'p':ps}

def save(obj,path):
    tmp=path.with_suffix('.pt.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def run(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if AMP:torch.cuda.manual_seed_all(args.seed)
    if not AMP:raise RuntimeError('CUDA required for this training configuration')
    tr=PairedWithQuality('train'); va=PairedWithQuality('val')
    train=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=collate,pin_memory=True)
    val=DataLoader(va,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate,pin_memory=True)
    model=FullTRCG().to(DEVICE)
    prefix=f'trcg_stage4b_phase{args.phase}_seed{args.seed}'
    best=CKPT/(prefix+'_best.pt');last=CKPT/(prefix+'_last.pt');done=CKPT/(prefix+'.done')
    hist=LOGS/(prefix+'_history.csv'); summary=RESULTS/(prefix+'_summary.json')
    if done.exists():print('Already done:',done);return
    if args.phase=='C':
        prev=CKPT/f'trcg_stage4b_phaseB_seed{args.seed}_best.pt'
        if not prev.exists():raise FileNotFoundError('Complete Phase B first: '+str(prev))
        model.load_state_dict(torch.load(prev,map_location='cpu',weights_only=True)['model'])
    mode(model,args.phase)
    groups=[{'params':[p for p in model.fusion.parameters() if p.requires_grad],'lr':args.fusion_lr},
            {'params':[p for p in model.ehr.parameters() if p.requires_grad],'lr':args.encoder_lr}]
    if args.phase=='C':groups.append({'params':[p for p in model.cxr.parameters() if p.requires_grad],'lr':args.encoder_lr})
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    npos=sum(int(r['label']) for r in tr.rows);pos_weight=torch.tensor((len(tr)-npos)/npos,device=DEVICE)
    start=1;bad=0;best_ap=-1.;best_ep=0
    if args.resume and last.exists():
        s=torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);scaler.load_state_dict(s['scaler'])
        start=s['epoch']+1;bad=s['bad'];best_ap=s['best_ap'];best_ep=s['best_ep']
        print('RESUMED after epoch',start-1)
    if args.smoke:
        batches(train,model,pos_weight,args.phase,optimizer,scaler,args.accum,smoke=True)
        print('SMOKE PASSED',args.phase);return
    for epoch in range(start,args.epochs+1):
        if bad>=args.patience:break
        train_res=batches(train,model,pos_weight,args.phase,optimizer,scaler,args.accum)
        val_res=batches(val,model,pos_weight,args.phase)
        improved=val_res['auprc']>best_ap+1e-6
        if improved:
            best_ap=val_res['auprc'];best_ep=epoch;bad=0
            save({'model':model.state_dict(),'epoch':epoch,'val_auroc':val_res['auroc'],'val_auprc':val_res['auprc']},best)
        else:bad+=1
        save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
              'epoch':epoch,'bad':bad,'best_ap':best_ap,'best_ep':best_ep},last)
        row={'epoch':epoch,'train_loss':train_res['loss'],'val_loss':val_res['loss'],
             'val_auroc':val_res['auroc'],'val_auprc':val_res['auprc'],'best_epoch':best_ep}
        with open(hist,'a',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(row));
            if f.tell()==0:w.writeheader()
            w.writerow(row)
        print(f'Phase {args.phase} epoch {epoch}: val AUROC={val_res["auroc"]:.4f} '
              f'AUPRC={val_res["auprc"]:.4f}; best={best_ep}; patience={bad}/{args.patience}',flush=True)
    if not best.exists():raise RuntimeError('No best model')
    ck=torch.load(best,map_location='cpu',weights_only=True)
    info={'phase':args.phase,'seed':args.seed,'best_epoch':ck['epoch'],
          'val_auroc':ck['val_auroc'],'val_auprc':ck['val_auprc'],'checkpoint':str(best),
          'note':'Validation-only selection. Test deliberately deferred until final phase selection.'}
    summary.write_text(json.dumps(info,indent=2),encoding='utf8')
    done.write_text(json.dumps(info,indent=2),encoding='utf8')
    print('PHASE COMPLETE:',json.dumps(info,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=['B','C'],required=True)
    a.add_argument('--epochs',type=int,default=12);a.add_argument('--patience',type=int,default=5)
    a.add_argument('--batch-size',type=int,default=2);a.add_argument('--accum',type=int,default=8)
    a.add_argument('--encoder-lr',type=float,default=1e-6);a.add_argument('--fusion-lr',type=float,default=1e-5)
    a.add_argument('--seed',type=int,default=42);a.add_argument('--resume',action='store_true');a.add_argument('--smoke',action='store_true')
    run(a.parse_args())
