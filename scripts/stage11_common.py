"""Shared safe utilities for Stage 11; all paths relative to TRCGFusion."""
from __future__ import annotations
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE, CKPT, RESULTS, LOGS = [ROOT / n for n in ('cache','checkpoints','results','logs')]
for folder in (CACHE, CKPT, RESULTS, LOGS): folder.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP = DEVICE.type == 'cuda'
SEEDS=(42,43,44)
FIELDS=('ehr_at_48','ehr_at_cxr','cxr','cxr_age','anchor_valid','ehr_quality','cxr_entropy')

def import_script(file, name):
    path = ROOT / 'scripts' / file
    if not path.is_file(): raise FileNotFoundError(f'Missing prior script: {path}')
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def atomic_torch(obj,path):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,temp); os.replace(temp,path)

def atomic_json(obj,path):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
    os.replace(temp,path)

def atomic_npz(path,**data):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    with open(temp,'wb') as f: np.savez_compressed(f,**data)
    os.replace(temp,path)

def atomic_csv(rows,path):
    if not rows:return
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(temp,path)

def paths(kind,seed):
    if kind not in ('ehr','cxr'):raise ValueError(kind)
    if seed == 42:
        return CKPT / f'trcg_stage2_{kind}_seed42_best.pt'
    return CKPT / f'trcg_stage11_{kind}_seed{seed}_best.pt'

def check_completed(done,settings):
    if not done.exists():return False
    entry=json.loads(done.read_text(encoding='utf8'))
    if entry.get('settings')!=settings:
        raise RuntimeError(f'Completed file has incompatible settings: {done}')
    return True

def read_feature(split,seed):
    path=CACHE/f'trcg_stage11_seed{seed}_{split}_features.npz'
    if not path.is_file():raise FileNotFoundError(f'Missing Stage11 cache {path}')
    with np.load(path,allow_pickle=False) as z:
        required=set(FIELDS+('stay_id','label','subject_id'))
        if required-set(z.files):raise RuntimeError(f'Missing keys in {path}: {required-set(z.files)}')
        d={k:np.asarray(z[k]).copy() for k in required}
    ids=d['stay_id'].astype(str); subjects=d['subject_id'].astype(str)
    y=d['label'].astype(np.int64)
    if not len(ids) or len(set(ids))!=len(ids):raise RuntimeError(f'Duplicate or empty stay IDs in {path}')
    if len(y)!=len(ids) or not np.isin(y,[0,1]).all():raise RuntimeError(f'Invalid labels in {path}')
    for key in FIELDS:
        a=np.asarray(d[key],dtype=np.float32)
        if len(a)!=len(y) or not np.isfinite(a).all():raise RuntimeError(f'Invalid {key}: {path}')
        d[key]=a
    if d['ehr_at_48'].shape != (len(y),128) or d['cxr'].shape!=(len(y),512):
        raise RuntimeError(f'Incorrect EHR or CXR dims: {path}')
    if d['ehr_quality'].shape!=(len(y),5):raise RuntimeError(f'Incorrect quality shape: {path}')
    d['stay_id']=ids; d['subject_id']=subjects; d['label']=y
    return d

def check_split_isolation(train,val):
    for field in ('stay_id','subject_id'):
        both=set(train[field]) & set(val[field])
        if both:raise RuntimeError(f'{field} leaks across train/validation: {list(both)[:3]}')
