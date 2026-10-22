TRCG-FUSION — STAGE 9: VALIDATION-ONLY MULTIMODAL CONTRIBUTION ANALYSIS

PURPOSE
Read-only comparison of Stage 7 safe_age, Stage 8 dense_fixed and separately
trained Stage 6 EHR-only (all seeds 42,43,44), matched by stay_id. The two
fusion models are also compared with the SAME original pretrained EHR anchor.
Measures actual changes to mortality logits, CXR-age/risk subgroups and paired
stratified bootstrap differences. No new model training. No test set opened.

INSTALL
Extract the ZIP into C:\MedModWork\TRCGFusion; this adds:
  scripts\25_stage9_contribution_analysis.py

REQUIRED PRIOR FILES
  scripts\18_train_stage7_refinement.py
  scripts\23_train_stage8_fusion.py
  src\models\trcg_fusion.py
  src\models\trcg_densenet_fusion.py
  cache\trcg_stage4_val_features.npz
  cache\trcg_stage8_val_features.npz
  checkpoints\trcg_stage2_ehr_seed42_best.pt
  checkpoints\trcg_stage2_cxr_seed42_best.pt
  checkpoints\trcg_stage3_align_seed42_best.pt
  checkpoints\trcg_stage6b_ehr_only_seed{42,43,44}_best.pt
  checkpoints\trcg_stage7_safe_age_seed{42,43,44}_best.pt
  checkpoints\trcg_stage8_dense_fixed_seed{42,43,44}_best.pt
  C:\MedModWork\MedModRepo\checkpoints_local\densenet121_best.pt

RUN — POWERSHELL
  cd C:\MedModWork\TRCGFusion
  & C:\MedModWork\MedMod\.venv\Scripts\Activate.ps1
  python -m py_compile .\scripts\25_stage9_contribution_analysis.py
  python .\scripts\25_stage9_contribution_analysis.py --smoke
  python .\scripts\25_stage9_contribution_analysis.py

OUTPUT FILES IN results\
  trcg_stage9_validation_predictions.csv
  trcg_stage9_model_comparison.csv
  trcg_stage9_paired_bootstrap.csv
  trcg_stage9_subgroup_analysis.csv
  trcg_stage9_summary.json

IMPORTANT INTERPRETATION
  - Stage 7/8 model variant choice already used validation scores.
  - Bootstrap confidence intervals are exploratory/descriptive only.
  - EHR-only trained ablation and frozen original EHR head are DIFFERENT.
  - Three fusion seeds reuse the same frozen single-seed encoders.
  - Avoid choosing final variants based on repeatedly viewed test results.

NO trained models, cached arrays, or older summaries are modified.
