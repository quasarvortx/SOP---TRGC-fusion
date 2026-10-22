"""Sequential, resumable Stage-4B Phase B -> Phase C."""
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts'/'12_train_trcg_endtoend.py'
CK=ROOT/'checkpoints'
RES=ROOT/'results'
for phase in ('B','C'):
    done=CK/f'trcg_stage4b_phase{phase}_seed42.done'
    if done.exists():
        print('Skipping complete phase',phase,flush=True)
        continue
    cmd=[sys.executable,'-u',str(SCRIPT),'--phase',phase,'--epochs','12','--patience','5',
         '--batch-size','2','--accum','8','--resume']
    print('Starting phase',phase,flush=True)
    ret=subprocess.run(cmd,cwd=ROOT,check=False)
    if not done.exists():
        raise SystemExit(f'Phase {phase} failed or interrupted (exit {ret.returncode}). Re-run this runner to resume.')
    print('Phase',phase,'complete',flush=True)

summaries=[]
for phase in ('B','C'):
    p=RES/f'trcg_stage4b_phase{phase}_seed42_summary.json'
    summaries.append(json.loads(p.read_text(encoding='utf8')))
chosen=max(summaries,key=lambda x:(x['val_auprc'],x['val_auroc']))
output={'comparison':summaries,'selected_phase':chosen['phase'],
        'selected_checkpoint':chosen['checkpoint'],'selection_basis':'validation AUPRC, then AUROC',
        'note':'Do not optimize selection using test outcomes. Stage 5 calibration/test still pending.'}
p=RES/'trcg_stage4b_selected_seed42_summary.json'
p.write_text(json.dumps(output,indent=2),encoding='utf8')
print(json.dumps(output,indent=2))
