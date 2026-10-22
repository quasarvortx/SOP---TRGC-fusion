
import subprocess
import sys

from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]

SCRIPT = (
    ROOT
    / "scripts"
    / "06_train_unimodal.py"
)

CHECKPOINTS = ROOT / "checkpoints"


def log(*parts):
    print(
        datetime.now().strftime(
            "[%Y-%m-%d %H:%M:%S]"
        ),
        *parts,
        flush=True,
    )


for model, batch_size in (
    ("ehr", 8),
    ("cxr", 4),
):
    marker = (
        CHECKPOINTS
        / f"trcg_stage2_{model}_seed42.done"
    )

    if marker.exists():
        log(
            "SKIP",
            model,
            "- already completed",
        )
        continue

    command = [
        sys.executable,
        "-u",
        str(SCRIPT),
        "--model", model,
        "--epochs", "35",
        "--patience", "8",
        "--batch-size", str(batch_size),
        "--seed", "42",
        "--resume",
    ]

    log(
        "START",
        model,
        "batch_size",
        batch_size,
    )

    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )

    # Success is determined using the completion marker,
    # not just the process exit code.
    if marker.exists():
        log(
            "FINISHED",
            model,
            "exit_code",
            result.returncode,
        )
        continue

    log(
        "FAILED",
        model,
        "exit_code",
        result.returncode,
        "No completion marker.",
    )

    sys.exit(1)


log(
    "ALL STAGE-2 RUNS COMPLETE.",
    "See results/trcg_stage2_*_summary.json",
)
