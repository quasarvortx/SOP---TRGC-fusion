TRCG Fusion - Stage 4A (frozen-encoder fusion ablation)
==================================================

This package adds THREE files to the existing C:\MedModWork\TRCGFusion repository:

src\models\trcg_fusion.py
scripts\10_train_trcg_frozen.py
scripts\11_run_stage4a.py

DO NOT replace your existing Stage 2/3 checkpoint files or script files.

Requirements (already produced by your previous stages):
  checkpoints\trcg_stage2_ehr_seed42_best.pt
  checkpoints\trcg_stage2_cxr_seed42_best.pt
  checkpoints\trcg_stage3_align_seed42_best.pt
  cache\trcg_manifest.csv
  cache\{train,val,test}_event_cache.pt
  cache\trcg_stage3_{train,val}_features.npz
  scripts\08_train_temporal_alignment.py

PowerShell:
  cd C:\MedModWork\TRCGFusion
  & C:\MedModWork\MedMod\.venv\Scripts\Activate.ps1
  python -m py_compile .\src\models\trcg_fusion.py .\scripts\10_train_trcg_frozen.py .\scripts\11_run_stage4a.py
  python .\scripts\10_train_trcg_frozen.py prepare
  python .\scripts\10_train_trcg_frozen.py smoke --variant v1
  python .\scripts\10_train_trcg_frozen.py smoke --variant v2
  python .\scripts\11_run_stage4a.py

The Stage 3 TEST encoder feature cache may be built during prepare (one-time
ResNet forward run). Validation selects v1 or v2 (AUPRC first), then only the
selected variant is evaluated on test.

Resume after reboot: run the SAME python .\scripts\11_run_stage4a.py command.
Completed runs are skipped; incomplete runs resume after the last checkpointed
full epoch. The *.done markers are saved after validation summary is complete.

Outputs:
  results\trcg_stage4a_v1_seed42_summary.json
  results\trcg_stage4a_v2_seed42_summary.json
  results\trcg_stage4a_selected_seed42_summary.json
  results\trcg_stage4a_selected_seed42_test_predictions.csv
  checkpoints\trcg_stage4a_{v1,v2}_seed42_{best,last}.pt

Science/methodology:
  V1: no Stage 3 projector initialization.
  V2: loads Stage 3 projector initialization.
  Both use cached frozen Stage 2 EHR and CXR features, reliability estimates,
  temporal concordance, clinical change, and gated residual correction.
  Encoder fine-tuning (Stage 4B) and calibration (Stage 5) are NOT performed.
  CXR image-quality feature is a learned representation proxy, not a validated
  image acquisition quality score. Complementarity regularizer is OFF by
  default (optional ablation). Follow-up end-to-end runs will be needed.
