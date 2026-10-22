import sys
from pathlib import Path

import pandas as pd


# ============================================================
# IMPORT PROJECT CONFIG
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
    EHR_ROOT,
    CXR_ROOT,
    LOG_DIR,
)


# ============================================================
# LOGGING
# ============================================================

OUTPUT = []


def log(*items):

    text = " ".join(
        str(x)
        for x in items
    )

    print(text)

    OUTPUT.append(text)


def section(title):

    log()
    log("=" * 80)
    log(title)
    log("=" * 80)


# ============================================================
# COLUMN HELPERS
# ============================================================

TIME_WORDS = [
    "time",
    "date",
    "hour",
    "intime",
    "outtime",
    "chart",
    "study",
    "cxr",
]

ID_WORDS = [
    "subject",
    "stay",
    "hadm",
    "dicom",
    "study",
]

PATH_WORDS = [
    "path",
    "file",
    "filename",
    "image",
    "cxr",
    "ehr",
]


def matching_columns(
    columns,
    keywords,
):

    result = []

    for col in columns:

        lower = str(col).lower()

        if any(
            word in lower
            for word in keywords
        ):

            result.append(col)

    return result


# ============================================================
# BASIC ENVIRONMENT
# ============================================================

section(
    "TRCG DATA AUDIT"
)

log(
    "EHR root:",
    EHR_ROOT,
)

log(
    "CXR root:",
    CXR_ROOT,
)


# ============================================================
# PAIRED CSV FILES
# ============================================================

for split_name, path in [

    ("TRAIN", PAIRED_TRAIN),

    ("VAL", PAIRED_VAL),

    ("TEST", PAIRED_TEST),

]:

    section(
        f"PAIRED {split_name}"
    )


    log(
        "Path:",
        path,
    )


    if not path.exists():

        log(
            "ERROR: file does not exist"
        )

        continue


    df = pd.read_csv(
        path
    )


    log(
        "Rows:",
        len(df),
    )


    log(
        "Columns:"
    )


    for col in df.columns:

        log(
            "  ",
            col,
            "| dtype =",
            df[col].dtype,
        )


    id_cols = matching_columns(
        df.columns,
        ID_WORDS,
    )


    time_cols = matching_columns(
        df.columns,
        TIME_WORDS,
    )


    path_cols = matching_columns(
        df.columns,
        PATH_WORDS,
    )


    log()
    log(
        "Possible ID columns:",
        id_cols,
    )


    log(
        "Possible TIME columns:",
        time_cols,
    )


    log(
        "Possible PATH columns:",
        path_cols,
    )


    interesting = list(
        dict.fromkeys(
            id_cols
            +
            time_cols
            +
            path_cols
        )
    )


    if interesting:

        log()
        log(
            "First 3 rows of relevant columns:"
        )


        preview = (
            df[
                interesting
            ]
            .head(3)
            .to_string(
                index=False
            )
        )


        for line in preview.splitlines():

            log(line)


# ============================================================
# EHR PROCESSED DIRECTORY
# ============================================================

section(
    "EHR DIRECTORY STRUCTURE"
)


if not EHR_ROOT.exists():

    log(
        "ERROR: EHR_ROOT missing"
    )

else:

    # Only top levels.
    for child in sorted(
        EHR_ROOT.iterdir()
    ):

        log(
            child.name,
            "DIR"
            if child.is_dir()
            else "FILE",
        )


# ============================================================
# LOCATE BENCHMARK LISTFILES
# ============================================================

section(
    "EHR LISTFILES"
)


listfiles = list(
    EHR_ROOT.rglob(
        "listfile.csv"
    )
)


log(
    "Number of listfile.csv files:",
    len(listfiles),
)


for listfile in listfiles:

    log()
    log(
        "Listfile:",
        listfile,
    )


    try:

        df = pd.read_csv(
            listfile
        )

    except Exception as exc:

        log(
            "Could not read:",
            exc,
        )

        continue


    log(
        "Columns:",
        list(df.columns),
    )


    log(
        "Rows:",
        len(df),
    )


    preview = (
        df.head(3)
        .to_string(
            index=False
        )
    )


    for line in preview.splitlines():

        log(line)


    # --------------------------------------------------------
    # Try to locate first referenced time-series file
    # --------------------------------------------------------

    if len(df) == 0:

        continue


    first_col = df.columns[0]

    candidate_name = str(
        df.iloc[0][first_col]
    )


    candidates = [

        listfile.parent
        / candidate_name,

        listfile.parent.parent
        / candidate_name,
    ]


    ts_path = None


    for candidate in candidates:

        if candidate.exists():

            ts_path = candidate

            break


    if ts_path is None:

        log(
            "Could not automatically locate referenced time-series file:",
            candidate_name,
        )

        continue


    log()
    log(
        "Example EHR time-series:",
        ts_path,
    )


    try:

        ts = pd.read_csv(
            ts_path
        )

    except Exception as exc:

        log(
            "Could not read time series:",
            exc,
        )

        continue


    log(
        "Shape:",
        ts.shape,
    )


    log(
        "Columns:"
    )


    for col in ts.columns:

        log(
            "  ",
            col,
        )


    log()
    log(
        "First 10 rows:"
    )


    preview = (
        ts.head(10)
        .to_string(
            index=False
        )
    )


    for line in preview.splitlines():

        log(line)


    possible_time = matching_columns(
        ts.columns,
        TIME_WORDS,
    )


    log()
    log(
        "Detected time columns:",
        possible_time,
    )


# ============================================================
# CHECK CXR METADATA FILES
# ============================================================

section(
    "CXR METADATA SEARCH"
)


metadata_candidates = []


for folder in [

    CXR_ROOT,

    CXR_ROOT.parent,

]:

    if not folder.exists():

        continue


    for pattern in [

        "*metadata*.csv",

        "*metadata*.csv.gz",

    ]:

        metadata_candidates.extend(

            folder.glob(
                pattern
            )
        )


metadata_candidates = list(
    dict.fromkeys(
        metadata_candidates
    )
)


log(
    "Metadata candidates found:",
    len(metadata_candidates),
)


for path in metadata_candidates:

    log()
    log(
        "Metadata:",
        path,
    )


    try:

        df = pd.read_csv(
            path,
            nrows=5,
        )

    except Exception as exc:

        log(
            "Could not read:",
            exc,
        )

        continue


    log(
        "Columns:"
    )


    for col in df.columns:

        log(
            "  ",
            col,
        )


    time_cols = matching_columns(
        df.columns,
        TIME_WORDS,
    )


    id_cols = matching_columns(
        df.columns,
        ID_WORDS,
    )


    log(
        "Possible timestamp columns:",
        time_cols,
    )


    log(
        "Possible ID columns:",
        id_cols,
    )


# ============================================================
# FINAL CHECKLIST
# ============================================================

section(
    "WHAT TRCG NEEDS"
)


required_items = [

    "subject_id",

    "stay_id",

    "mortality label",

    "ICU admission time",

    "raw EHR observation time",

    "EHR variable identity",

    "EHR observed value",

    "CXR path",

    "CXR acquisition datetime",

]


for item in required_items:

    log(
        "[CHECK]",
        item,
    )


# ============================================================
# SAVE LOG
# ============================================================

output_path = (
    LOG_DIR
    / "data_audit.txt"
)


output_path.write_text(

    "\n".join(
        OUTPUT
    ),

    encoding="utf-8",
)


section(
    "AUDIT COMPLETE"
)


log(
    "Saved full audit to:"
)

log(
    output_path
)