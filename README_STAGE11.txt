STAGE 11 — INDEPENDENT ENCODER SEEDS 42/43/44
================================================
Place scripts/* directly into C:\MedModWork\TRCGFusion\scripts (do not replace earlier files).

Dependencies from existing project:
- scripts/06_train_unimodal.py
- scripts/08_train_temporal_alignment.py
- scripts/18_train_stage7_refinement.py
- src/models/trcg_encoders.py
- src/models/trcg_fusion.py
- cache/trcg_manifest.csv, ehr_preprocessor.json, train_event_cache.pt, val_event_cache.pt
- cache/trcg_stage4_{train,val}_features.npz
- Stage2 seed42 EHR and CXR best checkpoints; Stage3 seed42 alignment best checkpoint

Commands in Windows PowerShell (existing .venv):
cd C:\MedModWork\TRCGFusion
& C:\MedModWork\MedMod\.venv\Scripts\Activate.ps1
python -m py_compile .\scripts\stage11_common.py .\scripts\27_stage11_train_encoders.py .\scripts\28_stage11_cache_features.py .\scripts\29_stage11_train_alignment.py .\scripts\30_stage11_train_fusion.py .\scripts\31_run_stage11.py
python .\scripts\31_run_stage11.py --smoke-only
python .\scripts\31_run_stage11.py

Re-run the last command to resume after an interruption. CXR training is expensive; leave
power plugged in and Windows sleep disabled.

Order:
1. Train independent EHR & ImageNet-initialized ResNet34 CXR seed43 and 44 (train/val only)
2. Cache aligned EHR at 48h, EHR at CXR, and CXR features for 42,43,44; recompute CXR entropy for 43 and 44
3. Align independently trained encoder features with separate Stage3 projectors, seed43 and 44
4. Train matched safe_age TRCG fusion against each exact EHR-only anchor, same encoder seed
5. Summarize mean +/- SD and validation-only, subject-cluster-bootstrap paired AUPRC delta

Important research caveats:
- The original EHR and CXR seed42 checkpoints are not retrained. Seeds 43/44 are trained with new initialization and data order, while all three CXR backbones start from the same ImageNet weights.
- Stage 11 is not an independently drawn cohort; this is run-level reproducibility across seeds.
- Stage 11 final validation is exploratory (we have repeatedly used these data). No TEST data read, no new calibration.
- Changing settings after a checkpoint is created is intentionally rejected. Use new names if you want a different experiment.
- No Stage 1–10 model or cache is overwritten.

Outputs:
results/trcg_stage11_validation_summary.json
results/trcg_stage11_per_seed_validation.csv
results/trcg_stage11_ensemble_validation_predictions.csv
and per-seed summaries, predictions, new checkpoints and logs.
