"""Resume-safe Stage-4A batch runner: feature cache -> V1 -> V2 -> select on validation -> test."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'scripts' / '10_train_trcg_frozen.py'
SEED = 42
CKPT, CACHE, RESULT = [ROOT / name for name in ('checkpoints', 'cache', 'results')]


def execute(label, args, success_paths):
    if all(p.exists() for p in success_paths):
        print('SKIP', label, '- outputs already exist', flush=True)
        return
    print('START', label, flush=True)
    command = [sys.executable, '-u', str(MAIN)] + args
    process = subprocess.run(command, cwd=ROOT, check=False)
    if all(p.exists() for p in success_paths):
        print('FINISHED', label, 'exit', process.returncode, flush=True)
        return
    raise SystemExit(f'FAILED {label}, exit={process.returncode}; outputs absent')


execute('prepare', ['prepare'], [CACHE / f'trcg_stage4_{s}_features.npz' for s in ('train', 'val', 'test')])
for variant in ('v1', 'v2'):
    execute(variant, ['train', '--variant', variant, '--seed', str(SEED), '--resume'],
            [CKPT / f'trcg_stage4a_{variant}_seed{SEED}.done',
             RESULT / f'trcg_stage4a_{variant}_seed{SEED}_summary.json'])
execute('select-test', ['select-test', '--seed', str(SEED)],
        [RESULT / f'trcg_stage4a_selected_seed{SEED}_summary.json',
         RESULT / f'trcg_stage4a_selected_seed{SEED}_test_predictions.csv'])
print('STAGE 4A FINISHED. Upload results/trcg_stage4a_selected_seed42_summary.json', flush=True)
