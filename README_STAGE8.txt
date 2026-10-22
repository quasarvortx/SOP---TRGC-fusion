TRCG-Fusion Stage 8 — reuse mortality-trained MedMod DenseNet121
==============================================================

Extract this ZIP into C:\MedModWork\TRCGFusion. It adds exactly:
 src\models\trcg_densenet_fusion.py
 scripts\22_cache_densenet_features.py
 scripts\23_train_stage8_fusion.py
 scripts\24_run_stage8.py

No MedModRepo files are modified, no original data extraction runs, and no test set is accessed.

IMPORTANT PREPROCESSING CHECK BEFORE RUNNING
--------------------------------------------
The checkpoint audit confirmed native DenseNet121 shapes and ImageNet-style normalization,
but could not conclusively establish whether your original train_densenet.py validation
pipeline resized the short edge to 256 and then center cropped 224, or resized the entire
image to 224x224. Inspect its data transform in the original repository (including functions
called from src/data/utils.py). Default is resize256_center224 as in prior TRCG stages.
Only proceed when you have confirmed the preprocessing mode matches training. If the
original model uses a different transform, adapt 22_cache_densenet_features.py first.

NOTE: also verify the DenseNet was trained on the same patient-level train split; otherwise
its frozen feature performance is not an independent comparison.

Commands (PowerShell):
 cd C:\MedModWork\TRCGFusion
 & C:\MedModWork\MedMod\.venv\Scripts\Activate.ps1
 python -m py_compile .\src\models\trcg_densenet_fusion.py .\scripts\22_cache_densenet_features.py .\scripts\23_train_stage8_fusion.py .\scripts\24_run_stage8.py
 python .\scripts\22_cache_densenet_features.py --split train --batch-size 2
 python .\scripts\22_cache_densenet_features.py --split val --batch-size 2
 python .\scripts\24_run_stage8.py --smoke-only --skip-cache
 python .\scripts\24_run_stage8.py --skip-cache

Output after training:
 results\trcg_stage8_all_validation_runs.csv
 results\trcg_stage8_validation_comparison.csv
 results\trcg_stage8_validation_summary.json

Resume: repeat exactly the last command; completed runs are skipped.
GPU OOM: decrease caching --batch-size to 1 (separate split caching commands).
The 3 fusion variants use fixed seed42 EHR+DenseNet encoders and seeds 42,43,44
for the FUSION HEAD only. They are not 3 independent end-to-end ensembles.

TEST DATA NOT ACCESSED. Evaluate final test after fixing selection rules and controls.
