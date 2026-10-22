"""Stage 8: cache original mortality-trained DenseNet121 1024-D CXR features.

Uses native checkpoint keys (vision_backbone.features.*, classifier.0.*).
Reads only TRAIN and VAL; test remains unopened.

python scripts/22_cache_densenet_features.py --split train --batch-size 2
python scripts/22_cache_densenet_features.py --split val --batch-size 2
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / 'cache'
OLD = Path(r'C:\MedModWork\MedModRepo')
DENSE_PT = OLD / 'checkpoints_local' / 'densenet121_best.pt'
MANIFEST = CACHE / 'trcg_manifest.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP = DEVICE.type == 'cuda'
TRANSFORMS = ('resize256_center224', 'resize224_square')

def atomic_npz(path, **data):
    tmp = path.with_suffix('.npz.partial')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **data)
    os.replace(tmp, path)

def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

class Images(Dataset):
    def __init__(self, split, mode):
        df = pd.read_csv(MANIFEST)
        self.rows = df.loc[df['split'] == split].reset_index(drop=True)
        if self.rows.empty:
            raise RuntimeError(f'No manifest rows for {split}')
        if self.rows['stay_id'].astype(str).duplicated().any():
            raise RuntimeError(f'Duplicate stay_id in {split}')
        if mode == 'resize256_center224':
            ops = [transforms.Resize(256), transforms.CenterCrop(224)]
        else:
            ops = [transforms.Resize((224, 224))]
        self.tf = transforms.Compose(ops + [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        with Image.open(str(row['cxr_path'])) as img:
            image = self.tf(img.convert('RGB'))
        return image, str(row['stay_id']), int(row['label'])

def load_densenet():
    # Original checkpoint is a plain state_dict, NOT nested under 'model'.
    if not DENSE_PT.exists():
        raise FileNotFoundError(DENSE_PT)
    sd = torch.load(DENSE_PT, map_location='cpu', weights_only=True)
    if not isinstance(sd, dict):
        raise RuntimeError('Expected checkpoint state_dict')
    features = {k[len('vision_backbone.features.'):]: v for k, v in sd.items()
                if k.startswith('vision_backbone.features.')}
    if not features:
        raise RuntimeError('No DenseNet features in checkpoint')
    net = models.densenet121(weights=None)
    # strict=True refuses incompatible/missing tensor weights.
    net.features.load_state_dict(features, strict=True)
    if sd.get('classifier.0.weight') is None or sd.get('classifier.0.bias') is None:
        raise RuntimeError('Original trained mortality classifier not found')
    w = sd['classifier.0.weight']
    bias = sd['classifier.0.bias']
    if tuple(w.shape) != (1, 1024) or tuple(bias.shape) != (1,):
        raise RuntimeError(f'Unexpected DenseNet classifier dimensions {w.shape}')
    net.classifier = torch.nn.Identity()
    net = net.to(DEVICE).eval()
    return net, w.to(DEVICE).float(), bias.to(DEVICE).float()

@torch.inference_mode()
def run(args):
    if args.split not in ('train', 'val'):
        raise ValueError('Test caching deliberately disabled during model selection')
    dataset = Images(args.split, args.transform)
    source = CACHE / f'trcg_stage4_{args.split}_features.npz'
    output = CACHE / f'trcg_stage8_{args.split}_features.npz'
    meta_path = CACHE / f'trcg_stage8_{args.split}_metadata.json'
    if not source.exists():
        raise FileNotFoundError(f'Run Stage 4A prepare first: {source}')
    info = dict(stage=8, split=args.split, checkpoint=str(DENSE_PT),
                checkpoint_sha256=hash_file(DENSE_PT), transform=args.transform,
                manifest=str(MANIFEST), source=str(source),
                source_size=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns,
                rows=len(dataset))
    if output.exists():
        if not meta_path.exists():
            raise RuntimeError(f'{output} exists without metadata; inspect or use --force')
        prior = json.loads(meta_path.read_text(encoding='utf-8'))
        if prior != info and not args.force:
            raise RuntimeError('Cached feature metadata differs; use --force if intentional')
        if prior == info and not args.force:
            print('REUSE (verified metadata):', output, flush=True)
            return
    net, w, bias = load_densenet()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=AMP)
    feature_list, id_list, label_list = [], [], []
    for images, ids, labels in tqdm(loader, desc=f'DenseNet {args.split}'):
        images = images.to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=AMP):
            fmap = net.features(images)
            z = F.adaptive_avg_pool2d(F.relu(fmap, inplace=False), (1, 1)).flatten(1)
        z = z.float()
        if z.shape[1] != 1024 or not torch.isfinite(z).all():
            raise RuntimeError('Invalid DenseNet features')
        feature_list.append(z.cpu().numpy())
        id_list.extend(ids)
        label_list.extend(labels.cpu().numpy().tolist())
    feat = np.concatenate(feature_list, axis=0).astype(np.float32)
    ids = np.asarray(id_list, dtype=str)
    labels = np.asarray(label_list, dtype=np.float32)
    if len(set(ids)) != len(ids):
        raise RuntimeError('Duplicate generated stay IDs')
    with np.load(source) as orig:
        old = {k: orig[k] for k in orig.files}
    index = {str(s): i for i, s in enumerate(ids)}
    orig_ids = [str(s) for s in old['stay_id']]
    if len(set(orig_ids)) != len(orig_ids) or set(orig_ids) != set(ids):
        raise RuntimeError(f'ID mismatch: Stage4 vs DenseNet {args.split}')
    order = np.asarray([index[s] for s in orig_ids], dtype=np.int64)
    if not np.array_equal(labels[order], old['label'].astype(np.float32)):
        raise RuntimeError('Label mismatch between DenseNet and EHR caches')
    feat = feat[order]
    logits = np.clip(feat @ w.cpu().numpy().reshape(-1) + float(bias.item()), -35, 35)
    prob = 1 / (1 + np.exp(-logits))
    entropy = - (prob*np.log(np.clip(prob, 1e-7, 1)) +
                 (1-prob)*np.log(np.clip(1-prob, 1e-7, 1))) / np.log(2.)
    result = {k: old[k] for k in (
        'ehr_at_48', 'ehr_at_cxr', 'cxr_age', 'anchor_valid',
        'ehr_quality', 'label', 'stay_id')}
    result.update(cxr=feat, cxr_entropy=entropy.astype(np.float32))
    if any(not np.isfinite(v).all() for k, v in result.items() if k != 'stay_id'):
        raise RuntimeError('Non-finite Stage8 feature values')
    atomic_npz(output, **result)
    meta_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'SAVED {output}: {len(ids)} stays, CXR shape={feat.shape}, mean entropy={entropy.mean():.4f}', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=('train', 'val'), required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--transform', choices=TRANSFORMS, default='resize256_center224')
    p.add_argument('--force', action='store_true')
    run(p.parse_args())
