TRCG Stage 6: validation-only diagnostics + frozen-feature ablations
====================================================================
Copy the ZIP CONTENTS into C:\MedModWork\TRCGFusion (preserves existing files).

Verify prerequisites:
  checkpoints\trcg_stage2_ehr_seed42_best.pt
  checkpoints\trcg_stage2_cxr_seed42_best.pt
  checkpoints\trcg_stage3_align_seed42_best.pt
  cache\trcg_stage4_train_features.npz
  cache\trcg_stage4_val_features.npz
  results\trcg_stage5_val_calibrated_seed42.csv

From activated venv:
  cd C:\MedModWork\TRCGFusion
  python -m py_compile scripts\15_stage6_diagnostics.py scripts\16_train_stage6_ablations.py scripts\17_run_stage6.py
  python scripts\17_run_stage6.py --smoke-only
  python scripts\17_run_stage6.py

Defaults: seeds 42,43,44 and variants full, ehr_only, no_reliability,
no_gate, no_change, no_align_loss, no_stage3_init; 12 maximum epochs, patience 5, lr 1e-4.
Resume after interruption with the SAME command.

Results:
  results\trcg_stage6a_diagnostics_seed42.json
  results\trcg_stage6b_validation_comparison.csv
  results\trcg_stage6b_all_validation_runs.csv

No test data or Stage 1-5 checkpoints are modified. This is NOT an
independent multiple-encoder-seed ensemble; Stage-2 encoders are frozen seed42
in every run. These are validation exploratory comparisons, not blind test
results. Checkpoint selection occurs on validation AUPRC.

Ablation definitions:
 full: Original Stage4A-v2 architecture and loss.
 ehr_only: EHR 48h features and mortality head only, trained separately.
 no_reliability: Force both learned reliability scalars to 1; keep CXR age in gate.
 no_gate: Force complementarity gamma to 1; retain r_CXR.
 no_change: Zero learned post-CXR clinical-change vector; retain other temporal cues.
 no_align_loss: Drop alignment loss, retain Stage3 initialization and architecture.
 no_stage3_init: Randomly initialize alignment projections, retaining alignment loss.

No claims of calibrated quality from raw AUPRC. Diagnostic EHR head of full
checkpoint and retrained EHR-only ablation serve different purposes.
