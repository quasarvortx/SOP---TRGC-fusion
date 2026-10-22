"""Run TRCG Stage 3 sequentially; restart with the same command after interruption."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / '08_train_temporal_alignment.py'
CACHE = ROOT / 'cache'
CKPT = ROOT / 'checkpoints'

steps = [
    ('Stage 3 feature caching', ['cache', '--cache-batch-size', '4'],
     [CACHE / 'trcg_stage3_train_features.npz', CACHE / 'trcg_stage3_val_features.npz']),
    ('Stage 3 alignment training', ['train', '--batch-size', '128', '--epochs', '50', '--patience', '8', '--resume'],
     [CKPT / 'trcg_stage3_align_seed42.done']),
]

for name, args, outputs in steps:
    if all(p.exists() for p in outputs):
        print(f'SKIP {name}: outputs already exist', flush=True)
        continue
    print(f'START {name}', flush=True)
    result = subprocess.run([sys.executable, '-u', str(SCRIPT), *args], cwd=ROOT)
    # Windows/PyTorch may sometimes crash during shutdown after saving results.
    if all(p.exists() for p in outputs):
        print(f'COMPLETE {name} (exit={result.returncode})', flush=True)
    else:
        print(f'STOPPED/FAILED {name} (exit={result.returncode})', flush=True)
        sys.exit(1)

print('TRCG STAGE 3 COMPLETE.', flush=True)
print('Results: results/trcg_stage3_align_seed42_summary.json', flush=True)
