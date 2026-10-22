"""Stage 5: validation-only temperature scaling of the best Stage 4A checkpoint.

Uses:
  - Stage 4A frozen val feature cache and v2/v1 checkpoint
  - Previous Stage 4A selected TEST prediction CSV (no retraining/reselection)

Outputs validation/test uncalibrated/calibrated predictions and metrics, with
reproducible bootstrap confidence intervals. Does not update model weights.

Run: python scripts/14_stage5_calibrate_evaluate.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"
SCRIPT = ROOT / "scripts" / "10_train_trcg_frozen.py"
RESULTS.mkdir(parents=True, exist_ok=True)


def load_stage4a_module():
    if not SCRIPT.exists():
        raise FileNotFoundError(f"Missing Stage 4A script: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("trcg_stage4a", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trcg_stage4a"] = mod
    spec.loader.exec_module(mod)
    return mod


def metrics(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1-1e-9)
    if len(np.unique(y)) != 2:
        raise ValueError("Metrics require both positive and negative samples")
    # Fixed-width 10-bin expected calibration error.
    bins = np.minimum(np.floor(p * 10).astype(int), 9)
    ece = 0.0
    for i in range(10):
        idx = bins == i
        if idx.any():
            ece += idx.mean() * abs(float(p[idx].mean()) - float(y[idx].mean()))
    return {
        "n": int(len(y)),
        "deaths": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10bins": float(ece),
    }


def bootstrap_auc_ci(y, p, seed=42, n_reps=1000):
    """Fixed-positive-count bootstrap avoids one-class resamples."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    auroc = np.empty(n_reps, dtype=np.float64)
    auprc = np.empty(n_reps, dtype=np.float64)
    for i in range(n_reps):
        idx = np.r_[rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)]
        auroc[i] = roc_auc_score(y[idx], p[idx])
        auprc[i] = average_precision_score(y[idx], p[idx])
    return {
        "method": "stratified bootstrap; fixed class counts",
        "replicates": n_reps,
        "seed": seed,
        "auroc_95ci": list(map(float, np.percentile(auroc, [2.5, 97.5]))),
        "auprc_95ci": list(map(float, np.percentile(auprc, [2.5, 97.5]))),
    }


def temperature_fit(logits, labels):
    """Fit a single positive T on validation unweighted NLL only."""
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("Invalid validation logits")
    def objective(log_t):
        z = x / math.exp(float(log_t))
        return float(np.mean(np.logaddexp(0, z) - y * z))
    res = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded",
                          options={"xatol": 1e-10})
    if not res.success:
        raise RuntimeError(f"Temperature fitting failed: {res.message}")
    temp = float(math.exp(float(res.x)))
    if not (math.isfinite(temp) and temp > 0):
        raise RuntimeError("Invalid fitted temperature")
    return temp


def validate_frame(df, name):
    for col in ("stay_id", "label", "probability"):
        if col not in df.columns:
            raise ValueError(f"{name} missing {col}")
    if len(df) == 0 or df.stay_id.astype(str).duplicated().any():
        raise ValueError(f"{name} empty or duplicate stay_id")
    if not df["label"].isin((0, 1)).all():
        raise ValueError(f"{name}: invalid mortality labels")
    p = df["probability"].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError(f"{name}: invalid probabilities")


def get_val_predictions(mod, checkpoint_path, variant, batch_size):
    model = mod.make_model(variant)
    # Load cached fixed checkpoint on the module's device; no parameter updates.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    dataset = mod.FeatureDataset("val")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []
    with torch.inference_mode():
        for batch, labels, ids in loader:
            batch = {key: val.to(mod.DEVICE) for key, val in batch.items()}
            o = model(batch)
            logit = o["logits"].double().cpu().numpy()
            ehr = o["ehr_logits"].double().cpu().numpy()
            cxr = o["cxr_logits"].double().cpu().numpy()
            ys = labels.int().cpu().numpy()
            for j, stay_id in enumerate(ids):
                rows.append({
                    "stay_id": str(stay_id),
                    "label": int(ys[j]),
                    "logit": float(logit[j]),
                    "probability": float(expit(logit[j])),
                    "ehr_probability": float(expit(ehr[j])),
                    "cxr_probability": float(expit(cxr[j])),
                    "gamma": float(o["gamma"][j]),
                    "r_ehr": float(o["r_ehr"][j]),
                    "r_cxr": float(o["r_cxr"][j]),
                })
    df = pd.DataFrame(rows)
    validate_frame(df, "validation")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-only", action="store_true",
                        help="Fit and report validation calibration; do not touch test")
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap < 1:
        parser.error("batch-size and bootstrap must be positive")

    selection_path = RESULTS / "trcg_stage4a_selected_seed42_summary.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing Stage 4A selection summary: {selection_path}")
    chosen = json.loads(selection_path.read_text(encoding="utf-8"))
    variant = chosen["selected_variant"]
    if variant not in ("v1", "v2"):
        raise ValueError(f"Unexpected chosen variant: {variant}")
    checkpoint = CKPT / f"trcg_stage4a_{variant}_seed42_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    checkpoint_info = torch.load(checkpoint, map_location="cpu", weights_only=True)
    selected_epoch = int(checkpoint_info["epoch"])
    mod = load_stage4a_module()
    val = get_val_predictions(mod, checkpoint, variant, args.batch_size)
    val_logit = val["logit"].to_numpy(dtype=np.float64)
    temp = temperature_fit(val_logit, val.label.to_numpy())
    val["calibrated_probability"] = expit(val_logit / temp)
    val_path = RESULTS / "trcg_stage5_val_calibrated_seed42.csv"
    val.to_csv(val_path, index=False)
    val_raw = metrics(val.label, val.probability)
    val_cal = metrics(val.label, val.calibrated_probability)
    ehr_val = metrics(val.label, val.ehr_probability)

    report = {
        "stage": 5,
        "selection": "Stage 4A vs 4B compared on validation only; Stage 4A remains best",
        "selected_model": f"Stage4A-{variant} epoch{selected_epoch} frozen encoders",
        "selected_checkpoint": str(checkpoint),
        "temperature_fit": "unweighted validation negative log-likelihood",
        "temperature": temp,
        "validation": {"uncalibrated": val_raw, "calibrated": val_cal,
                       "ehr_head_uncalibrated": ehr_val},
        "validation_file": str(val_path),
        "caveats": [
            "Single seed; not a five-member ensemble.",
            "Temperature scaling is strictly monotone; AUROC/AUPRC should not change.",
            "Stage4A test results were observed previously; this is not a fresh blind test.",
            "Confidence intervals describe sampling variation, not model-selection uncertainty.",
        ],
    }
    print(f"Checkpoint: Stage4A-{variant}")
    print(f"Validation: raw AUPRC={val_raw['auprc']:.4f}; "
          f"calibrated NLL={val_cal['log_loss']:.4f}; temperature={temp:.4f}")

    if not args.val_only:
        test_path = RESULTS / "trcg_stage4a_selected_seed42_test_predictions.csv"
        if not test_path.exists():
            raise FileNotFoundError(
                "Missing existing Stage4A TEST predictions: " + str(test_path))
        test = pd.read_csv(test_path, dtype={"stay_id": str})
        validate_frame(test, "test")
        # DO NOT fit or retune temperature with test data.
        p = np.clip(test.probability.to_numpy(dtype=np.float64), 1e-9, 1-1e-9)
        logit = np.log(p) - np.log1p(-p)
        test["calibrated_probability"] = expit(logit / temp)
        test_out = RESULTS / "trcg_stage5_test_calibrated_seed42.csv"
        test.to_csv(test_out, index=False)
        raw_m = metrics(test.label, test.probability)
        cal_m = metrics(test.label, test.calibrated_probability)
        report["test"] = {
            "uncalibrated": raw_m,
            "calibrated": cal_m,
            "calibrated_bootstrap_95ci": bootstrap_auc_ci(
                test.label.to_numpy(), test.calibrated_probability.to_numpy(),
                seed=args.seed, n_reps=args.bootstrap),
        }
        report["test_file"] = str(test_out)
        print(f"Test AUROC={cal_m['auroc']:.4f}; AUPRC={cal_m['auprc']:.4f}; "
              f"Brier before={raw_m['brier']:.4f}; after={cal_m['brier']:.4f}")
        assert np.isclose(raw_m["auroc"], cal_m["auroc"], atol=1e-10)
        assert np.isclose(raw_m["auprc"], cal_m["auprc"], atol=1e-10)
        if not np.isclose(raw_m["auroc"], chosen["test_auroc"], atol=1e-6):
            raise RuntimeError("Stage4A test AUC differs from selected summary")
        if not np.isclose(raw_m["auprc"], chosen["test_auprc"], atol=1e-6):
            raise RuntimeError("Stage4A test AUPRC differs from selected summary")

    report_path = RESULTS / "trcg_stage5_seed42_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Summary:", report_path)
    print("No model weights were updated. All temperature fitting used validation only.")


if __name__ == "__main__":
    main()
