import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    r"C:\MedModWork\TRCGFusion"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from config import (
    CACHE_DIR,
    LOG_DIR,
)


MANIFEST_PATH = (
    CACHE_DIR
    / "trcg_manifest.csv"
)


VARIABLES = [

    "Capillary refill rate",
    "Diastolic blood pressure",
    "Fraction inspired oxygen",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale total",
    "Glascow coma scale verbal response",
    "Glucose",
    "Heart Rate",
    "Height",
    "Mean blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Systolic blood pressure",
    "Temperature",
    "Weight",
    "pH",
]


# GCS components are strings in MIMIC.
CATEGORICAL = {
    "Capillary refill rate",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale verbal response",
}


manifest = pd.read_csv(
    MANIFEST_PATH
)


train_manifest = manifest[
    manifest["split"] == "train"
].copy()


print(
    "Training stays:",
    len(train_manifest)
)


# ============================================================
# COLLECT VALUES FROM TRAIN ONLY
# ============================================================

numeric_values = {
    v: []
    for v in VARIABLES
    if v not in CATEGORICAL
}


categorical_values = {
    v: set()
    for v in VARIABLES
    if v in CATEGORICAL
}


for i, row in train_manifest.iterrows():

    if i % 500 == 0:
        print(
            f"Processing {i}/{len(train_manifest)}"
        )

    df = pd.read_csv(
        row["ehr_path"]
    )

    df["Hours"] = pd.to_numeric(
        df["Hours"],
        errors="coerce",
    )

    df = df[
        (df["Hours"] >= 0)
        &
        (df["Hours"] <= 48)
    ]


    for variable in VARIABLES:

        series = df[variable].dropna()


        if variable in CATEGORICAL:

            for value in series:

                categorical_values[
                    variable
                ].add(
                    str(value)
                    .strip()
                )


        else:

            values = pd.to_numeric(
                series,
                errors="coerce",
            ).dropna()

            numeric_values[
                variable
            ].extend(
                values.tolist()
            )


# ============================================================
# NUMERIC STATISTICS
# ============================================================

numeric_stats = {}


for variable, values in numeric_values.items():

    arr = np.asarray(
        values,
        dtype=np.float64,
    )


    if len(arr) == 0:

        mean = 0.0
        std = 1.0
        count = 0

    else:

        mean = float(
            np.mean(arr)
        )

        std = float(
            np.std(arr)
        )

        count = int(
            len(arr)
        )

        if std < 1e-8:
            std = 1.0


    numeric_stats[
        variable
    ] = {
        "mean": mean,
        "std": std,
        "count": count,
    }


# ============================================================
# CATEGORICAL VOCABULARIES
#
# 0 = missing / unknown
# category indices begin at 1
# ============================================================

categorical_vocab = {}


for variable, values in categorical_values.items():

    values = sorted(
        list(values)
    )


    vocab = {
        value: index + 1
        for index, value
        in enumerate(values)
    }


    categorical_vocab[
        variable
    ] = vocab


# ============================================================
# SAVE
# ============================================================

output = {

    "variables":
        VARIABLES,

    "categorical_variables":
        sorted(
            list(
                CATEGORICAL
            )
        ),

    "numeric_stats":
        numeric_stats,

    "categorical_vocab":
        categorical_vocab,

    "normalization":
        "train_only_zscore",

    "observation_window":
        48.0,
}


output_path = (
    CACHE_DIR
    / "ehr_preprocessor.json"
)


with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

report = []


report.append(
    "EHR PREPROCESSOR REPORT"
)

report.append(
    "=" * 70
)


for variable in VARIABLES:

    if variable in CATEGORICAL:

        vocab = categorical_vocab[
            variable
        ]

        report.append(
            f"{variable}: "
            f"CATEGORICAL | "
            f"{len(vocab)} categories"
        )

        for key, value in vocab.items():

            report.append(
                f"    {value}: {key}"
            )

    else:

        stats = numeric_stats[
            variable
        ]

        report.append(

            f"{variable}: "
            f"NUMERIC | "
            f"n={stats['count']} | "
            f"mean={stats['mean']:.6f} | "
            f"std={stats['std']:.6f}"
        )


report_path = (
    LOG_DIR
    / "ehr_preprocessor_report.txt"
)


report_path.write_text(

    "\n".join(
        report
    ),

    encoding="utf-8",
)


print()
print(
    "Saved:"
)

print(
    output_path
)

print(
    report_path
)