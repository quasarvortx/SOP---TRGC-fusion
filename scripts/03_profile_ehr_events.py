import sys
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
    /
    "trcg_manifest.csv"
)


manifest = pd.read_csv(
    MANIFEST_PATH
)


report = []


def log(*items):

    text = " ".join(
        str(x)
        for x in items
    )

    print(text)

    report.append(text)


# ============================================================
# EXPECTED EHR VARIABLES
# ============================================================

ehr_variables = [

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


log(
    "Manifest rows:",
    len(
        manifest
    )
)


event_counts = []

row_counts = []

last_times = []

before_cxr_counts = []

after_cxr_counts = []

missing_by_variable = {
    var: 0
    for var in ehr_variables
}

observed_by_variable = {
    var: 0
    for var in ehr_variables
}


bad_files = []

leak_rows = 0


# ============================================================
# PROCESS EVERY STAY
# ============================================================

for idx, row in manifest.iterrows():

    ehr_path = Path(
        row[
            "ehr_path"
        ]
    )


    try:

        df = pd.read_csv(
            ehr_path
        )

    except Exception as exc:

        bad_files.append(
            (
                str(
                    ehr_path
                ),
                str(
                    exc
                )
            )
        )

        continue


    # --------------------------------------------------------
    # Enforce 48-hour prediction window
    # --------------------------------------------------------

    df[
        "Hours"
    ] = pd.to_numeric(
        df[
            "Hours"
        ],
        errors="coerce",
    )


    df = df[
        df[
            "Hours"
        ].notna()
    ].copy()


    # Any raw rows >48h?
    if (
        df[
            "Hours"
        ]
        >
        48.0
    ).any():

        leak_rows += int(

            (
                df[
                    "Hours"
                ]
                >
                48.0
            ).sum()
        )


    # Keep leakage-safe portion only
    df = df[

        (
            df[
                "Hours"
            ]
            >= 0
        )

        &

        (
            df[
                "Hours"
            ]
            <=
            48.0
        )

    ].copy()


    # --------------------------------------------------------
    # Number of unique timestamp rows
    # --------------------------------------------------------

    row_counts.append(
        len(
            df
        )
    )


    if len(
        df
    ) > 0:

        last_times.append(

            float(
                df[
                    "Hours"
                ].max()
            )
        )


    # --------------------------------------------------------
    # Count actual observed events
    #
    # If 3 variables are measured at the same timestamp,
    # that is 3 events.
    # --------------------------------------------------------

    observed_matrix = (
        df[
            ehr_variables
        ]
        .notna()
    )


    total_events = int(
        observed_matrix
        .sum()
        .sum()
    )


    event_counts.append(
        total_events
    )


    # --------------------------------------------------------
    # Before / after CXR
    # --------------------------------------------------------

    cxr_hour = float(
        row[
            "cxr_hour"
        ]
    )


    before = df[
        df[
            "Hours"
        ]
        <=
        cxr_hour
    ]


    after = df[
        (
            df[
                "Hours"
            ]
            >
            cxr_hour
        )
        &
        (
            df[
                "Hours"
            ]
            <=
            48.0
        )
    ]


    before_events = int(

        before[
            ehr_variables
        ]
        .notna()
        .sum()
        .sum()
    )


    after_events = int(

        after[
            ehr_variables
        ]
        .notna()
        .sum()
        .sum()
    )


    before_cxr_counts.append(
        before_events
    )


    after_cxr_counts.append(
        after_events
    )


    # --------------------------------------------------------
    # Variable observation statistics
    # --------------------------------------------------------

    for var in ehr_variables:

        observed = int(
            df[
                var
            ]
            .notna()
            .sum()
        )


        observed_by_variable[
            var
        ] += observed


        missing_by_variable[
            var
        ] += (
            len(df)
            -
            observed
        )


# ============================================================
# HELPER
# ============================================================

def describe(
    values,
    name,
):

    x = np.asarray(
        values,
        dtype=float,
    )


    log()
    log(
        name
    )


    log(
        "  n      =",
        len(x)
    )


    log(
        "  mean   =",
        round(
            np.mean(x),
            2
        )
    )


    log(
        "  median =",
        round(
            np.median(x),
            2
        )
    )


    log(
        "  p90    =",
        round(
            np.percentile(
                x,
                90
            ),
            2
        )
    )


    log(
        "  p95    =",
        round(
            np.percentile(
                x,
                95
            ),
            2
        )
    )


    log(
        "  p99    =",
        round(
            np.percentile(
                x,
                99
            ),
            2
        )
    )


    log(
        "  max    =",
        round(
            np.max(x),
            2
        )
    )


# ============================================================
# REPORT
# ============================================================

log()
log("=" * 70)
log("EHR EVENT PROFILE")
log("=" * 70)


describe(
    row_counts,
    "Timestamp rows per patient"
)


describe(
    event_counts,
    "Observed variable events per patient"
)


describe(
    before_cxr_counts,
    "Observed events BEFORE/AT CXR"
)


describe(
    after_cxr_counts,
    "Observed events AFTER CXR until 48h"
)


describe(
    last_times,
    "Last available EHR time"
)


log()
log(
    "Raw >48h rows encountered:",
    leak_rows
)


log()
log(
    "Bad EHR files:",
    len(
        bad_files
    )
)


# ============================================================
# VARIABLE COUNTS
# ============================================================

log()
log("=" * 70)
log("VARIABLE OBSERVATION COUNTS")
log("=" * 70)


for var in ehr_variables:

    log(
        var,
        ":",
        observed_by_variable[
            var
        ]
    )


# ============================================================
# SAVE
# ============================================================

output = (
    LOG_DIR
    /
    "ehr_event_profile.txt"
)


output.write_text(

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
    output
)