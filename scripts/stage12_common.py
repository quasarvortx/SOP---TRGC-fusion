"""Stage 12 additions: use existing Stage 11 utilities and seed-specific paths."""
from stage11_common import *  # reuse all Stage 11 safe read/write utilities
from stage11_common import ROOT, CACHE, CKPT, RESULTS, LOGS
from pathlib import Path
import numpy as np

ALL_SEEDS = (42,43,44,45,46)
NEW_SEEDS = (45,46)

def paths(kind, seed):
    if kind not in ('ehr','cxr') or seed not in ALL_SEEDS:
        raise ValueError((kind,seed))
    if seed == 42:
        return CKPT / f'trcg_stage2_{kind}_seed42_best.pt'
    if seed in (43,44):
        return CKPT / f'trcg_stage11_{kind}_seed{seed}_best.pt'
    return CKPT / f'trcg_stage12_{kind}_seed{seed}_best.pt'

def align_path(seed):
    if seed == 42:
        return CKPT / 'trcg_stage3_align_seed42_best.pt'
    if seed in (43,44):
        return CKPT / f'trcg_stage11_align_seed{seed}_best.pt'
    if seed in NEW_SEEDS:
        return CKPT / f'trcg_stage12_align_seed{seed}_best.pt'
    raise ValueError(seed)

def fusion_path(seed):
    if seed in (42,43,44):
        return CKPT / f'trcg_stage11_safe_age_seed{seed}_best.pt'
    if seed in NEW_SEEDS:
        return CKPT / f'trcg_stage12_safe_age_seed{seed}_best.pt'
    raise ValueError(seed)

def feature_path(split,seed):
    if seed not in ALL_SEEDS or split not in ('train','val','test'):
        raise ValueError((split,seed))
    if split=='test' or seed in NEW_SEEDS:
        return CACHE / f'trcg_stage12_seed{seed}_{split}_features.npz'
    return CACHE / f'trcg_stage11_seed{seed}_{split}_features.npz'

def read_feature(split,seed):
    path=feature_path(split,seed)
    if not path.exists(): raise FileNotFoundError(path)
    from stage11_common import FIELDS
    with np.load(path,allow_pickle=False) as z:
        required=set(FIELDS+('stay_id','label','subject_id','source_seed'))
        if required-set(z.files):
            raise RuntimeError(f'Missing fields from {path}: {required-set(z.files)}')
        if int(z['source_seed'].reshape(-1)[0])!=seed:
            raise RuntimeError(f'Seed provenance mismatch in {path}')
        d={k:np.asarray(z[k]).copy() for k in required}
    d['stay_id']=d['stay_id'].astype(str)
    d['subject_id']=d['subject_id'].astype(str)
    d['label']=d['label'].astype(int)
    ids=d['stay_id'];n=len(ids)
    if n==0 or len(set(ids))!=n or len(d['subject_id'])!=n:
        raise RuntimeError(f'Invalid stay/subject IDs in {path}')
    if not np.isin(d['label'],[0,1]).all():raise RuntimeError(f'Invalid labels in {path}')
    for key in FIELDS:
        a=np.asarray(d[key],dtype=np.float32)
        if len(a)!=n or not np.isfinite(a).all(): raise RuntimeError(f'Invalid feature {key} in {path}')
        d[key]=a
    if d['ehr_at_48'].shape!=(n,128) or d['ehr_at_cxr'].shape!=(n,128) or d['cxr'].shape!=(n,512):
        raise RuntimeError(f'Wrong feature dimension in {path}')
    if d['ehr_quality'].shape!=(n,5):raise RuntimeError(f'Wrong quality descriptor size in {path}')
    return d
