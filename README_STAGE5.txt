TRCG Stage 5 — validation-only calibration and reporting
=========================================================

Copy scripts/14_stage5_calibrate_evaluate.py into
C:\MedModWork\TRCGFusion\scripts\

Do not change or overwrite any old model checkpoint.

Run in your activated Python venv:
    cd C:\MedModWork\TRCGFusion
    python -m py_compile .\scripts\14_stage5_calibrate_evaluate.py
    python .\scripts\14_stage5_calibrate_evaluate.py --val-only
    python .\scripts\14_stage5_calibrate_evaluate.py

Uses fixed Stage4A-v2 best checkpoint and the existing Stage4A test
predictions. Fits a single temperature on validation negative log-loss only.
No model retraining or test-based tuning. Caution: earlier Stage4A test result
was already observed, so this is not a wholly untouched holdout analysis.

Writes:
  results/trcg_stage5_val_calibrated_seed42.csv
  results/trcg_stage5_test_calibrated_seed42.csv
  results/trcg_stage5_seed42_summary.json

Key methodological caveats:
- Scaling probabilities with a positive temperature does not change their
  rank, so AUROC and AUPRC remain identical.
- A model trained with class-weighted BCE can have a shifted calibration
  intercept. Temperature-only scaling may not fully fix this.
- For strong publication-style results, rerun several independent seeds,
  include ablations, and keep a fresh patient-disjoint holdout truly blind.
