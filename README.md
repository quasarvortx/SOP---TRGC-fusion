# TRCG-Fusion

TRCG-Fusion studies whether chest X-rays can add useful information to an EHR model for in-hospital mortality prediction. The main question is how to use an image taken earlier in an ICU stay without treating it as a current measurement. The model combines temporal alignment, modality reliability estimates, and a gated correction to an EHR prediction made at 48 hours.

## Data

The project uses MIMIC-IV and MIMIC-CXR, with processed EHR time series and paired chest X-rays from the existing MedMod pipeline. The fixed patient-separated split contains 4,956 training stays, 490 validation stays, and 1,231 test stays. The test split contains 180 deaths.

MIMIC data require approved access and are not distributed here. Cohort manifests, identifiers, patient-level predictions, images, feature caches, and trained checkpoints are also excluded. The `results/` folder contains selected aggregate results from the completed experiments.

## Architecture

- **EHR encoder:** a causal, two-layer Transformer with 128-dimensional states and four attention heads. It represents 17 clinical variables using values, observation masks, time since observation, categorical embeddings, and continuous time embeddings.
- **CXR encoder:** an ImageNet-initialized ResNet34 with a 512-dimensional image representation and a mortality head. Stage 8 separately investigates a previously trained DenseNet121.
- **Temporal alignment:** project the EHR state available at the image time and the CXR representation into a shared 128-dimensional space. Contrastive training encourages agreement between paired representations.
- **Reliability and fusion:** use EHR quality descriptors, image features, image age, and CXR prediction entropy to estimate modality reliability. A learned gate controls a residual correction to the EHR state at 48 hours.

The original `TRCGFusion` includes a learned clinical-change branch. The final model is `SafeTRCG` with the `safe_age` setting in `scripts/18_train_stage7_refinement.py`: the learned change branch is disabled, the mortality heads are frozen, the image residual starts at zero, and an invalid pre-image EHR anchor suppresses the correction. A weak age-by-EHR-change penalty discourages reliance on stale images. This distinction matters when comparing the original proposal with the final evaluated model.

## Repository layout

```text
config.py                 Original data paths and Stage 1 settings
src/models/               EHR/CXR encoders and fusion modules
scripts/                  Numbered scripts 01–38 and shared Stage 11/12 utilities
notebooks/                Architecture and evaluation walkthrough
results/                  Reviewed aggregate metrics only
README_STAGE*.txt         Original notes for individual experiment stages
requirements*.txt         Research and notebook dependencies
```

## Setup

The completed runs used Python 3.14.2, PyTorch 2.11.0 with CUDA 12.8, and torchvision 0.26.0. The research package versions in `requirements.txt` were read from the local experiment environment; this is not a lockfile for every transitive dependency.

```powershell
git clone https://github.com/quasarvortx/SOP---TRGC-fusion.git
cd SOP---TRGC-fusion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-notebook.txt
python -m jupyterlab notebooks/architecture_and_evaluation.ipynb
```

The notebook's default cells review the actual model code and saved aggregate results without MIMIC data or checkpoints. Its optional inference section uses existing local checkpoints and cached features and is disabled by default.

Full reproduction needs the original processed data and paired split files, plus the local ImageNet ResNet34 weights expected by `06_train_unimodal.py`. Stage 8 also needs the existing MedMod DenseNet121 checkpoint. These inputs are not downloaded by this repository.

The original scripts are preserved, including their Windows path assumptions. Before running data-dependent scripts on another machine, update `config.py` and the `PROJECT_ROOT` constants in scripts `01`–`05`. Review the MedMod paths in `21_audit_stage8_cxr.py`, `22_cache_densenet_features.py`, and `23_train_stage8_fusion.py`; script `20` accepts `--paired-root` and `--cxr-root`. Later stages generally resolve the project root from their script location. The original working location was `C:\MedModWork\TRCGFusion`, with paired files in `C:\MedModWork\paired` and processed EHR data in `C:\MedModWork\ehr_processed\in-hospital-mortality`.

## Running the scripts

Run commands from the repository root. These are the existing experiment entry points, not steps needed to read the results. Training and evaluation commands need the private inputs and preceding stage outputs. The original stage notes give individual options and resume behavior.

| Stage | Work completed | Entry points |
| --- | --- | --- |
| 1 | Audit paired data, build the temporal manifest, fit EHR preprocessing on training data, cache events | `01_audit_existing_data.py` through `05_build_event_cache.py`, in order |
| 2 | Train EHR-only and ResNet34 CXR mortality encoders | `07_run_stage2.py` |
| 3 | Cache causal EHR/CXR representations and train temporal alignment | `09_run_stage3.py` |
| 4A / 4B | Frozen-feature fusion and staged encoder fine-tuning | `11_run_stage4a.py`, `13_run_stage4b.py` |
| 5 | Validation temperature scaling and evaluation | `14_stage5_calibrate_evaluate.py` |
| 6 | Reliability diagnostics and component ablations | `15_stage6_diagnostics.py`, `17_run_stage6.py` |
| 7 | EHR-anchored fusion with learned, fixed, and age-regularized gates | `19_run_stage7.py` |
| 8 | Frozen DenseNet121 comparison | `21_audit_stage8_cxr.py`, `24_run_stage8.py` |
| 9 | Contribution analysis and paired bootstrap comparisons | `25_stage9_contribution_analysis.py` |
| 10 | CXR embedding shuffle controls on validation data | `26_stage10_image_shuffle.py` |
| 11 | Independent encoder training for seeds 43–44, alongside seed 42 | `31_run_stage11.py` |
| 12 | Add seeds 45–46, ensemble all five seeds, lock calibration, evaluate test | `37_run_stage12.py`, `38_run_stage12_final_test.py` |

For example, after completing the earlier stages and restoring their local inputs:

```powershell
# Train/validate the additional Stage 12 seeds; this can be expensive.
python scripts/37_run_stage12.py --smoke-only
python scripts/37_run_stage12.py

# Separate fixed-model test evaluation after validation calibration is locked.
python scripts/38_run_stage12_final_test.py
```

Stage 12 reuses earlier checkpoints and caches. Its runner does not recreate Stages 1–11. Several earlier runners also evaluate the test split, so rerunning the full historical sequence is not a new blind evaluation. Generated patient-level outputs must remain local.

## Final results

Stage 12 averages predicted probabilities from seeds 42–46 and applies temperatures fitted on validation data. These are **calibrated five-seed ensemble test metrics**, not the mean of the individual seed metrics.

| Metric | EHR-only | TRCG-Fusion |
| --- | --- | --- |
| AUROC | 0.8131 | 0.8134 |
| AUPRC | 0.4459 | 0.4464 |
| Brier score | 0.1109 | 0.1085 |
| Log loss | 0.3608 | 0.3536 |

Fusion slightly improved probabilistic accuracy but did not demonstrate a statistically reliable improvement in AUROC or AUPRC. The paired subject-cluster bootstrap 95% intervals for fusion minus EHR were [−0.00245, 0.00384] for AUPRC and [−0.00051, 0.00106] for AUROC; both include zero. The Brier difference interval was [−0.00319, −0.00163], where lower is better.

Full-precision results, individual seed metrics, calibration temperatures, and bootstrap intervals are in `results/trcg_stage12_final_test_summary.json`. The notebook reads these files directly.

## Limitations

The validation cohort was repeatedly inspected during development, and the benchmark test set had been inspected in earlier stages. The final evaluation is retrospective rather than a fresh blind test. The five training seeds share the same patient split and ImageNet initialization; they do not establish generalization across hospitals or datasets. Bootstrap intervals measure patient-resampling uncertainty, not model-selection or dataset-shift uncertainty. The observed image contribution is small, and the experiments do not establish clinical utility. Full reproduction requires authorized MIMIC access and the original processed split and local model inputs.
