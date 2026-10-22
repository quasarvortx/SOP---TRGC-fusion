STAGE 10 — Read-only CXR pairing permutation diagnostic
========================================================

1) Extract into C:\MedModWork\TRCGFusion so scripts/26_stage10_image_shuffle.py exists.
   Leave all previous scripts, checkpoints and feature caches untouched.

2) Activate your existing Python venv:
   cd C:\MedModWork\TRCGFusion
   & C:\MedModWork\MedMod\.venv\Scripts\Activate.ps1

3) Smoke test:
   python -m py_compile .\scripts\26_stage10_image_shuffle.py
   python .\scripts\26_stage10_image_shuffle.py --smoke

4) Run:
   python .\scripts\26_stage10_image_shuffle.py --permutations 100

Stage 10 loads Stage 7 and 8 validation feature caches and selected fusion
checkpoints for seeds 42/43/44 via Stage 9's previously tested loaders.
It NEVER opens test data and NEVER changes weights.

Experiments:
 A) shuffle CXR embeddings and corresponding CXR entropy among all patients
 B) shuffle only within equally sized CXR-age quartiles

The recipient patient's clinical EHR state, age, valid anchor flag and EHR
quality features remain unchanged. Losing performance after shuffling supports
the model using patient-specific CXR pairing. Unchanged performance means that
fusion may primarily act as a calibration or risk-intercept adjustment.

Results:
 results/trcg_stage10_shuffle_summary.json
 results/trcg_stage10_shuffle_comparison.csv
 results/trcg_stage10_shuffle_trials.csv
 results/trcg_stage10_patient_sensitivity.csv

Send summary.json and shuffle_comparison.csv after completion.

Note: the prior validation set was also used for model selection, so these are
exploratory negative controls, not a blind external confirmation.
