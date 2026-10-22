from pathlib import Path


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(
    r"C:\MedModWork\TRCGFusion"
)

MEDMOD_REPO = Path(
    r"C:\MedModWork\MedModRepo"
)


# ============================================================
# EXISTING DATA
# ============================================================

PAIRED_ROOT = Path(
    r"C:\MedModWork\paired"
)

PAIRED_TRAIN = (
    PAIRED_ROOT
    / "paired_train.csv"
)

PAIRED_VAL = (
    PAIRED_ROOT
    / "paired_val.csv"
)

PAIRED_TEST = (
    PAIRED_ROOT
    / "paired_test.csv"
)


EHR_ROOT = Path(
    r"C:\MedModWork\ehr_processed\in-hospital-mortality"
)


CXR_ROOT = Path(
    r"S:\MedModData\mimic-cxr-jpg\2.1.0"
)


# ============================================================
# EXISTING MEDMOD OBJECTS
# ============================================================

PAIRED_NORMALIZER = (
    MEDMOD_REPO
    / "normalizer_v31.pkl"
)


# ============================================================
# TRCG SETTINGS
# ============================================================

OBSERVATION_HOURS = 48

NUM_CLINICAL_VARIABLES = 17

SEED = 42


# ============================================================
# NEW PROJECT OUTPUTS
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "cache"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)


for folder in [
    CHECKPOINT_DIR,
    RESULT_DIR,
    CACHE_DIR,
    LOG_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )