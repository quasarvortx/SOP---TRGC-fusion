STAGE 12 — Five Independent Encoder Seeds + Frozen Validation Calibration
=======================================================================
This package extends your successful Stage 11 run. Extract it into
C:\MedModWork\TRCGFusion (MERGE folders). No Stage 1–11 file is overwritten.

Requires existing Stage 11 scripts: stage11_common.py, 27..31 scripts,
plus Stage 3, 6 and 7 dependencies, the paired dataset and all existing files.

STEP 1: PowerShell in C:\MedModWork\TRCGFusion
  & C:\MedModWork\MedMod\.venv\Scripts\Activate.ps1
  python -m py_compile .\scripts\32_stage12_train_encoders.py .\scripts\33_stage12_cache_features.py .\scripts\34_stage12_train_alignment.py .\scripts\35_stage12_train_fusion.py .\scripts\36_stage12_finalize.py .\scripts\37_run_stage12.py .\scripts\38_run_stage12_final_test.py .\scripts\stage12_common.py

STEP 2: GPU smoke tests (no training checkpoints overwritten)
  python .\scripts\37_run_stage12.py --smoke-only

STEP 3: One command for unattended TWO NEW SEEDS (45,46), caches,
alignment, fusion and five-seed validation-only calibration.
  python .\scripts\37_run_stage12.py

Output after Step 3:
  results\trcg_stage12_validation_summary.json
  results\trcg_stage12_five_seed_validation_predictions.csv
  results\trcg_stage12_locked_temperatures.json

Review and share validation summary before final test.

STEP 4: Explicitly run FINAL TEST ONLY after configuration is locked:
  python .\scripts\38_run_stage12_final_test.py

This command builds per-seed test features from existing processed files
and applies validation-fitted temperatures unchanged. It does not train,
change model selection, or tune hyperparameters on the test set.

Output:
  results\trcg_stage12_final_test_summary.json
  results\trcg_stage12_per_seed_test.csv
  results\trcg_stage12_five_seed_test_predictions.csv

Important cautions:
- The benchmark test set was already inspected earlier in your project;
  final evaluation is RETROSPECTIVE, NOT a new blind test.
- Five training seeds use the same patient split and ImageNet initialization.
- CXR images must use the SAME preprocessing as Stage 11.
- For test caching, existing cache\trcg_stage4_test_features.npz and
  cache\test_event_cache.pt are required. Re-run only the original Stage4A
  feature PREPARATION step if the former is missing; never repeat extraction.
- Resume support: rerun the same 37 command after interruption.
- To reduce GPU memory, use --cxr-batch 1 consistently from start of run.
  Changing batch-size mid-run will trigger a resume-settings mismatch.
- Stop immediately if any check reports nonfinite features, wrong IDs,
  mismatched seed, changed validation predictions, or missing checkpoints.
