import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(
    r"C:\MedModWork\TRCGFusion"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from config import (
    PAIRED_TRAIN,
    PAIRED_VAL,
    PAIRED_TEST,
    CACHE_DIR,
    LOG_DIR,
    OBSERVATION_HOURS,
)


# ============================================================
# SETTINGS
# ============================================================

PREDICTION_HOUR = float(
    OBSERVATION_HOURS
)


# ============================================================
# OUTPUT
# ============================================================

output_rows = []

report = []


def log(*items):

    text = " ".join(
        str(x)
        for x in items
    )

    print(text)

    report.append(text)


# ============================================================
# PROCESS SPLIT
# ============================================================

for split_name, csv_path in [

    ("train", PAIRED_TRAIN),

    ("val", PAIRED_VAL),

    ("test", PAIRED_TEST),

]:

    log()
    log("=" * 70)
    log("PROCESSING", split_name.upper())
    log("=" * 70)

    df = pd.read_csv(
        csv_path
    )

    log(
        "Rows:",
        len(df)
    )


    # --------------------------------------------------------
    # Parse datetimes
    # --------------------------------------------------------

    df["intime"] = pd.to_datetime(
        df["intime"],
        errors="coerce",
    )

    df["StudyDateTime"] = pd.to_datetime(
        df["StudyDateTime"],
        errors="coerce",
    )


    # --------------------------------------------------------
    # Recalculate CXR relative hour
    #
    # We DO NOT blindly trust time_diff.
    # We recompute it from timestamps and compare.
    # --------------------------------------------------------

    df[
        "cxr_hour_recomputed"
    ] = (

        (
            df["StudyDateTime"]
            -
            df["intime"]
        )

        .dt.total_seconds()

        / 3600.0
    )


    df[
        "cxr_hour"
    ] = df[
        "time_diff"
    ].astype(float)


    df[
        "cxr_time_error"
    ] = (

        df["cxr_hour"]

        -

        df[
            "cxr_hour_recomputed"
        ]

    ).abs()


    # --------------------------------------------------------
    # Prediction time = 48 hours
    # --------------------------------------------------------

    df[
        "prediction_hour"
    ] = PREDICTION_HOUR


    # --------------------------------------------------------
    # Image age
    # --------------------------------------------------------

    df[
        "cxr_age"
    ] = (

        PREDICTION_HOUR

        -

        df["cxr_hour"]
    )


    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    df[
        "cxr_before_admission"
    ] = (

        df["cxr_hour"] < 0
    )


    df[
        "cxr_after_prediction"
    ] = (

        df["cxr_hour"]
        >
        PREDICTION_HOUR
    )


    df[
        "missing_datetime"
    ] = (

        df["intime"].isna()

        |

        df[
            "StudyDateTime"
        ].isna()
    )


    df[
        "ehr_exists"
    ] = (

        df["ehr_path"]
        .astype(str)
        .map(
            lambda x:
            Path(x).exists()
        )
    )


    df[
        "cxr_exists"
    ] = (

        df["cxr_path"]
        .astype(str)
        .map(
            lambda x:
            Path(x).exists()
        )
    )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    log(
        "CXR hour range:",
        round(
            df["cxr_hour"].min(),
            3
        ),
        "to",
        round(
            df["cxr_hour"].max(),
            3
        ),
    )


    log(
        "CXR age range:",
        round(
            df["cxr_age"].min(),
            3
        ),
        "to",
        round(
            df["cxr_age"].max(),
            3
        ),
    )


    log(
        "CXR before admission:",
        int(
            df[
                "cxr_before_admission"
            ].sum()
        ),
    )


    log(
        "CXR after 48h:",
        int(
            df[
                "cxr_after_prediction"
            ].sum()
        ),
    )


    log(
        "Missing datetime:",
        int(
            df[
                "missing_datetime"
            ].sum()
        ),
    )


    log(
        "Missing EHR files:",
        int(
            (
                ~df["ehr_exists"]
            ).sum()
        ),
    )


    log(
        "Missing CXR files:",
        int(
            (
                ~df["cxr_exists"]
            ).sum()
        ),
    )


    log(
        "Max difference between "
        "stored time_diff and recomputed time:",
        round(
            df[
                "cxr_time_error"
            ].max(),
            6
        ),
        "hours",
    )


    # --------------------------------------------------------
    # Only retain leakage-safe examples
    # --------------------------------------------------------

    valid = df[

        (~df[
            "cxr_before_admission"
        ])

        &

        (~df[
            "cxr_after_prediction"
        ])

        &

        (~df[
            "missing_datetime"
        ])

        &

        (df[
            "ehr_exists"
        ])

        &

        (df[
            "cxr_exists"
        ])

    ].copy()


    log(
        "VALID:",
        len(valid),
        "/",
        len(df),
    )


    # --------------------------------------------------------
    # Compact manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame({

        "subject_id":
            valid[
                "subject_id"
            ].astype(int),

        "stay_id":
            valid[
                "stay_id"
            ].astype(int),

        "dicom_id":
            valid[
                "dicom_id"
            ].astype(str),

        "study_id":
            valid[
                "study_id"
            ].astype(int),

        "split":
            split_name,

        "label":
            valid[
                "y_true"
            ].astype(int),

        "icu_intime":
            valid[
                "intime"
            ].astype(str),

        "cxr_datetime":
            valid[
                "StudyDateTime"
            ].astype(str),

        "cxr_hour":
            valid[
                "cxr_hour"
            ].astype(float),

        "prediction_hour":
            PREDICTION_HOUR,

        "cxr_age":
            valid[
                "cxr_age"
            ].astype(float),

        "ehr_path":
            valid[
                "ehr_path"
            ].astype(str),

        "cxr_path":
            valid[
                "cxr_path"
            ].astype(str),
    })


    output_rows.append(
        manifest
    )


# ============================================================
# CONCATENATE
# ============================================================

manifest = pd.concat(
    output_rows,
    ignore_index=True,
)


# ============================================================
# PATIENT SPLIT LEAKAGE CHECK
# ============================================================

log()
log("=" * 70)
log("PATIENT SPLIT LEAKAGE CHECK")
log("=" * 70)


patient_splits = (

    manifest

    .groupby(
        "subject_id"
    )[
        "split"
    ]

    .nunique()
)


leaking_subjects = (
    patient_splits
    [
        patient_splits > 1
    ]
)


log(
    "Subjects appearing in multiple splits:",
    len(
        leaking_subjects
    ),
)


if len(
    leaking_subjects
) > 0:

    log(
        "WARNING:"
    )

    log(
        leaking_subjects.head(
            20
        )
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

log()
log("=" * 70)
log("CLASS DISTRIBUTION")
log("=" * 70)


for split in [
    "train",
    "val",
    "test",
]:

    part = manifest[
        manifest[
            "split"
        ]
        ==
        split
    ]


    deaths = int(
        part[
            "label"
        ].sum()
    )


    total = len(part)


    log(
        split,
        "| total =",
        total,
        "| deaths =",
        deaths,
        "| prevalence =",
        round(
            deaths
            /
            total,
            4,
        ),
    )


# ============================================================
# CXR AGE DISTRIBUTION
# ============================================================

log()
log("=" * 70)
log("CXR AGE DISTRIBUTION")
log("=" * 70)


for split in [
    "train",
    "val",
    "test",
]:

    part = manifest[
        manifest[
            "split"
        ]
        ==
        split
    ]


    log(
        split,
        "| median age =",
        round(
            part[
                "cxr_age"
            ].median(),
            3,
        ),
        "| mean age =",
        round(
            part[
                "cxr_age"
            ].mean(),
            3,
        ),
        "| min =",
        round(
            part[
                "cxr_age"
            ].min(),
            3,
        ),
        "| max =",
        round(
            part[
                "cxr_age"
            ].max(),
            3,
        ),
    )


# ============================================================
# SAVE
# ============================================================

output_path = (
    CACHE_DIR
    /
    "trcg_manifest.csv"
)


manifest.to_csv(
    output_path,
    index=False,
)


report_path = (
    LOG_DIR
    /
    "manifest_report.txt"
)


report_path.write_text(
    "\n".join(
        report
    ),
    encoding="utf-8",
)


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Manifest:"
)

print(
    output_path
)

print()

print(
    "Report:"
)

print(
    report_path
)