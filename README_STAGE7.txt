TRCG Stage 7: minimal, controlled fusion refinement (validation only)
====================================================================

PREREQUISITES
All Stage 2/3/4A checkpoints and Stage 4 train/val feature caches.
Do NOT delete or modify any Stage 1–6 results.

EXTRACT ZIP INTO
C:\MedModWork\TRCGFusion\

ONE-TIME COHORT AUDIT (READ-ONLY)
  python .\scripts\20_audit_cxr_cohort.py

This verifies:
  - selected images are AP and within ICU and <=48h;
  - patient split disjointness;
  - selected image is the most recent eligible AP among metadata candidates.
The earlier benchmark cohort sizes differ; use this audit before another study.
If the audit needs review, fix the cohort methodology separately and do not
claim exact MedMod comparability. Do not casually replace the current cohort.

SMOKE TEST
  python .\scripts\19_run_stage7.py --smoke-only

TRAIN/RESUME (ALL 9 RUNS)
  python .\scripts\19_run_stage7.py

This runs three variants x seeds 42/43/44 with frozen encoder features:
  safe_learned: zero-initialized CXR residual; pretrained EHR head frozen;
                no learned EHR-change module and no alignment objective;
                learned patient-specific gate.
  safe_fixed: same setup but a fixed gate set to the MEAN original Stage-4A
              gate on TRAIN, a correctly magnitude-matched no-gate control.
  safe_age: same as safe_learned, plus weak age x observed EHR-change
            regularization as proposed (lambda 0.02).
All variants force CXR residual weight to 0 for no valid pre-CXR EHR anchor.

The script checks that fused logits EQUAL pretrained EHR logits BEFORE updates.
This is a data-safe refinement study, not a guarantee of better performance.
It reads train/val only and will NOT evaluate on test.

FILES
results/trcg_stage7_all_validation_runs.csv
results/trcg_stage7_validation_comparison.csv
results/trcg_stage7_*_summary.json
checkpoints/trcg_stage7_*_best.pt, *_last.pt, *.done
logs/trcg_stage7_*_history.csv
logs/stage7_cxr_cohort_audit.txt

Send back:
  logs/stage7_cxr_cohort_audit.txt
  results/trcg_stage7_validation_comparison.csv
  results/trcg_stage7_all_validation_runs.csv

Caveats: all seeds reuse same pretrained seed-42 EHR/CXR encoders;
we have already examined the test split previously; validation sample only
490. Avoid winner's curse and grand performance claims after many variants.
