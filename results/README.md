# Aggregate experiment results

These files were copied unchanged from the completed local runs. No experiments were rerun to prepare this repository.

- Stages 6–8: validation comparisons for ablations and fusion variants. These stages varied fusion-head seeds with shared encoders.
- Stages 9–10: validation contribution comparisons, paired bootstrap results, and image shuffle controls.
- Stage 11: validation summaries for independently trained encoder seeds 42–44.
- Stage 12: validation summary, locked validation-fitted temperatures, final retrospective test summary, and per-seed test metrics for seeds 42–46.

The final test JSON contains calibrated and uncalibrated ensemble metrics. Its `per_seed` entries and the corresponding CSV are uncalibrated individual models; they should not be confused with calibrated ensemble results. The bootstrap intervals compare calibrated fusion and EHR ensemble predictions, with resampling clustered by subject.

Only aggregate metrics and experiment metadata are included. Predictions, identifiers, cohort manifests, caches, and subgroup or patient-level exports remain local. The `.gitignore` uses an explicit allowlist for this folder; review the contents of any new artifact before adding it.
