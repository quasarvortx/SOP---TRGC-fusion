import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(
    r"C:\MedModWork\TRCGFusion"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from config import (
    CACHE_DIR,
)


MANIFEST_PATH = (
    CACHE_DIR
    / "trcg_manifest.csv"
)


PREPROCESSOR_PATH = (
    CACHE_DIR
    / "ehr_preprocessor.json"
)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    PREPROCESSOR_PATH,
    "r",
    encoding="utf-8",
) as f:

    PREP = json.load(f)


VARIABLES = PREP[
    "variables"
]


CATEGORICAL = set(
    PREP[
        "categorical_variables"
    ]
)


NUMERIC_STATS = PREP[
    "numeric_stats"
]


CATEGORY_VOCAB = PREP[
    "categorical_vocab"
]


N_VARIABLES = len(
    VARIABLES
)


manifest = pd.read_csv(
    MANIFEST_PATH
)


# ============================================================
# NORMALIZATION
# ============================================================

def encode_numeric(
    variable,
    value,
):

    if pd.isna(value):

        return 0.0


    x = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]


    if pd.isna(x):

        return 0.0


    mean = NUMERIC_STATS[
        variable
    ]["mean"]


    std = NUMERIC_STATS[
        variable
    ]["std"]


    value = (
        float(x)
        -
        mean
    ) / std


    # Robust clipping prevents extreme outliers
    # destabilizing early Transformer training.

    value = float(
        np.clip(
            value,
            -10.0,
            10.0,
        )
    )


    return value


def encode_category(
    variable,
    value,
):

    if pd.isna(value):

        return 0


    value = str(
        value
    ).strip()


    return int(
        CATEGORY_VOCAB[
            variable
        ].get(
            value,
            0,
        )
    )


# ============================================================
# BUILD ONE STAY
# ============================================================

def build_stay(
    row,
):

    df = pd.read_csv(
        row["ehr_path"]
    )


    df["Hours"] = pd.to_numeric(
        df["Hours"],
        errors="coerce",
    )


    df = df[
        df["Hours"].notna()
    ]


    df = df[
        (df["Hours"] >= 0.0)
        &
        (df["Hours"] <= 48.0)
    ].copy()


    df = df.sort_values(
        "Hours"
    )


    # --------------------------------------------------------
    # If multiple original rows happen to have identical time,
    # combine them by taking the last non-null observation
    # for each variable.
    # --------------------------------------------------------

    rows = []


    for timestamp, group in df.groupby(
        "Hours",
        sort=True,
    ):

        combined = {
            "Hours": float(
                timestamp
            )
        }


        for variable in VARIABLES:

            observed = (
                group[
                    variable
                ]
                .dropna()
            )


            if len(observed) > 0:

                combined[
                    variable
                ] = observed.iloc[-1]

            else:

                combined[
                    variable
                ] = np.nan


        rows.append(
            combined
        )


    df = pd.DataFrame(
        rows
    )


    seq_len = len(
        df
    )


    # ========================================================
    # OUTPUT CHANNELS
    #
    # numerical_values: [T,17]
    # categorical_ids:  [T,17]
    # mask:             [T,17]
    # delta:            [T,17]
    # times:            [T]
    # ========================================================

    numerical_values = np.zeros(
        (
            seq_len,
            N_VARIABLES,
        ),
        dtype=np.float32,
    )


    categorical_ids = np.zeros(
        (
            seq_len,
            N_VARIABLES,
        ),
        dtype=np.int64,
    )


    mask = np.zeros(
        (
            seq_len,
            N_VARIABLES,
        ),
        dtype=np.float32,
    )


    delta = np.zeros(
        (
            seq_len,
            N_VARIABLES,
        ),
        dtype=np.float32,
    )


    times = (
        df["Hours"]
        .to_numpy(
            dtype=np.float32
        )
    )


    last_observed = np.full(
        N_VARIABLES,
        np.nan,
        dtype=np.float32,
    )


    # ========================================================
    # BUILD TOKENS
    # ========================================================

    for t_index in range(
        seq_len
    ):

        current_time = float(
            times[
                t_index
            ]
        )


        for var_index, variable in enumerate(
            VARIABLES
        ):

            value = df.iloc[
                t_index
            ][variable]


            observed = not pd.isna(
                value
            )


            if observed:

                mask[
                    t_index,
                    var_index
                ] = 1.0


                if variable in CATEGORICAL:

                    categorical_ids[
                        t_index,
                        var_index
                    ] = encode_category(
                        variable,
                        value,
                    )

                else:

                    numerical_values[
                        t_index,
                        var_index
                    ] = encode_numeric(
                        variable,
                        value,
                    )


            # ------------------------------------------------
            # Time since PREVIOUS measurement
            #
            # If variable was never measured previously,
            # delta = elapsed time since admission.
            # ------------------------------------------------

            if np.isnan(
                last_observed[
                    var_index
                ]
            ):

                d = current_time

            else:

                d = (
                    current_time
                    -
                    float(
                        last_observed[
                            var_index
                        ]
                    )
                )


            d = np.clip(
                d,
                0.0,
                48.0,
            )


            # Normalize into approximately [0,1]
            delta[
                t_index,
                var_index
            ] = (
                d / 48.0
            )


            # Update timestamp AFTER calculating delta
            if observed:

                last_observed[
                    var_index
                ] = current_time


    # Normalize absolute time.
    normalized_time = (
        times / 48.0
    ).astype(
        np.float32
    )


    return {

        "subject_id":
            int(
                row[
                    "subject_id"
                ]
            ),

        "stay_id":
            int(
                row[
                    "stay_id"
                ]
            ),

        "label":
            int(
                row[
                    "label"
                ]
            ),

        "cxr_hour":
            float(
                row[
                    "cxr_hour"
                ]
            ),

        "cxr_age":
            float(
                row[
                    "cxr_age"
                ]
            ),

        "cxr_path":
            str(
                row[
                    "cxr_path"
                ]
            ),

        "times":
            torch.tensor(
                times,
                dtype=torch.float32,
            ),

        "normalized_time":
            torch.tensor(
                normalized_time,
                dtype=torch.float32,
            ),

        "values":
            torch.tensor(
                numerical_values,
                dtype=torch.float32,
            ),

        "categories":
            torch.tensor(
                categorical_ids,
                dtype=torch.long,
            ),

        "mask":
            torch.tensor(
                mask,
                dtype=torch.float32,
            ),

        "delta":
            torch.tensor(
                delta,
                dtype=torch.float32,
            ),
    }


# ============================================================
# BUILD EACH SPLIT
# ============================================================

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
    ].reset_index(
        drop=True
    )


    print()
    print("=" * 70)
    print(
        "BUILDING",
        split.upper()
    )
    print("=" * 70)


    samples = []


    for _, row in tqdm(
        part.iterrows(),
        total=len(part),
    ):

        sample = build_stay(
            row
        )

        samples.append(
            sample
        )


    output_path = (

        CACHE_DIR
        /
        f"{split}_event_cache.pt"
    )


    torch.save(
        samples,
        output_path,
    )


    lengths = [
        len(
            sample[
                "times"
            ]
        )
        for sample in samples
    ]


    print(
        "Patients:",
        len(samples)
    )


    print(
        "Mean sequence:",
        np.mean(
            lengths
        )
    )


    print(
        "Maximum sequence:",
        np.max(
            lengths
        )
    )


    print(
        "Saved:"
    )

    print(
        output_path
    )