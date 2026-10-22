
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torchvision import transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.trcg_encoders import (
    EHRMortality,
    CXRMortality,
)


CACHE = ROOT / "cache"
CKPT = ROOT / "checkpoints"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"

for folder in (CKPT, LOGS, RESULTS):
    folder.mkdir(parents=True, exist_ok=True)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
AMP = DEVICE.type == "cuda"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if AMP:
        torch.cuda.manual_seed_all(seed)


def atomic_save(payload, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


# ============================================================
# EHR DATASET
# ============================================================

class EHREventDataset(Dataset):
    def __init__(self, split):
        p = CACHE / f"{split}_event_cache.pt"

        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. "
                "Run scripts/05_build_event_cache.py first."
            )

        # Locally generated trusted event cache.
        self.items = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        if not self.items:
            raise ValueError(f"Empty event cache: {p}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def ehr_collate(samples):
    lengths = torch.tensor(
        [len(s["times"]) for s in samples],
        dtype=torch.long,
    )

    if (lengths <= 0).any():
        raise ValueError("Empty EHR sequence found")

    B = len(samples)
    T = int(lengths.max())
    V = 17

    batch = {
        "values": torch.zeros(B, T, V),
        "categories": torch.zeros(
            B, T, V, dtype=torch.long
        ),
        "mask": torch.zeros(B, T, V),
        "delta": torch.zeros(B, T, V),
        "times": torch.zeros(B, T),
        "lengths": lengths,
    }

    for i, s in enumerate(samples):
        L = len(s["times"])
        t = s["times"].float()

        if (
            not torch.isfinite(t).all()
            or (t < 0).any()
            or (t > 48.00001).any()
        ):
            raise ValueError(
                f"Invalid event hours: {s['stay_id']}"
            )

        if (t[1:] < t[:-1]).any():
            raise ValueError(
                f"Unsorted event hours: {s['stay_id']}"
            )

        for key in (
            "values",
            "categories",
            "mask",
            "delta",
        ):
            if s[key].shape != (L, V):
                raise ValueError(
                    f"Invalid {key} shape: {s['stay_id']}"
                )

            batch[key][i, :L] = s[key]

        batch["times"][i, :L] = t

    labels = torch.tensor([
        float(s["label"]) for s in samples
    ])

    ids = [
        str(s["stay_id"]) for s in samples
    ]

    return batch, labels, ids


# ============================================================
# CXR DATASET
# ============================================================

class CXRDataset(Dataset):
    def __init__(self, split):
        manifest_path = CACHE / "trcg_manifest.csv"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}"
            )

        self.rows = pd.read_csv(manifest_path)

        self.rows = self.rows[
            self.rows["split"] == split
        ].reset_index(drop=True)

        if self.rows.empty:
            raise ValueError(
                f"No CXR rows for {split}"
            )

        if (
            (self.rows["cxr_hour"] < 0)
            | (self.rows["cxr_hour"] > 48)
        ).any():
            raise ValueError(
                "CXR timestamp outside [0,48]h"
            )

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if split == "train":
            steps = [
                transforms.Resize(256),
                transforms.RandomAffine(
                    degrees=7,
                    translate=(0.025, 0.025),
                    scale=(0.97, 1.03),
                ),
                transforms.CenterCrop(224),
            ]
        else:
            steps = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]

        self.tf = transforms.Compose(
            steps + [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows.iloc[index]

        with Image.open(str(r["cxr_path"])) as im:
            x = self.tf(im.convert("RGB"))

        y = torch.tensor(
            float(r["label"]),
            dtype=torch.float32,
        )

        return x, y, str(r["stay_id"])


# ============================================================
# HELPERS
# ============================================================

def to_device(x):
    if isinstance(x, dict):
        return {
            k: v.to(DEVICE, non_blocking=True)
            for k, v in x.items()
        }

    return x.to(DEVICE, non_blocking=True)


def make_dataloaders(model_name, batch_size):
    dataset_cls = (
        EHREventDataset
        if model_name == "ehr"
        else CXRDataset
    )

    collate = (
        ehr_collate
        if model_name == "ehr"
        else None
    )

    outputs = []

    for split in ("train", "val", "test"):
        ds = dataset_cls(split)

        outputs.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=0,
                pin_memory=AMP,
                collate_fn=collate,
            )
        )

        print(f"{split}: {len(ds)} examples")

    return outputs


def make_model(model_name):
    if model_name == "ehr":
        prep_path = (
            CACHE / "ehr_preprocessor.json"
        )

        if not prep_path.exists():
            raise FileNotFoundError(
                f"Missing {prep_path}"
            )

        prep = json.loads(
            prep_path.read_text(encoding="utf-8")
        )

        if len(prep["variables"]) != 17:
            raise ValueError(
                "Expected exactly 17 EHR variables"
            )

        return EHRMortality(
            prep["categorical_vocab"]
        )

    checkpoint = (
        Path.home()
        / ".cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "resnet34-b627a593.pth"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing cached ResNet-34 weights:\n"
            f"{checkpoint}\n"
            "Do not silently use random initialization."
        )

    return CXRMortality(checkpoint)


# ============================================================
# VALIDATION AND TESTING
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    ys = []
    probs = []
    stay_ids = []

    loss_total = 0.0
    total = 0

    for x, y, ids in loader:
        x = to_device(x)
        y = y.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(
            "cuda",
            enabled=AMP,
        ):
            logits = model(x)

        loss = criterion(
            logits.float(),
            y.float(),
        )

        p = torch.sigmoid(
            logits.float()
        )

        loss_total += loss.item() * len(y)
        total += len(y)

        ys.extend(
            y.cpu().numpy().tolist()
        )

        probs.extend(
            p.cpu().numpy().tolist()
        )

        stay_ids.extend(ids)

    auc = float(
        roc_auc_score(ys, probs)
    )

    auprc = float(
        average_precision_score(ys, probs)
    )

    return {
        "loss": loss_total / total,
        "auroc": auc,
        "auprc": auprc,
        "labels": ys,
        "probabilities": probs,
        "stay_ids": stay_ids,
    }


# ============================================================
# TRAIN
# ============================================================

def run(args):
    seed_everything(args.seed)

    train_loader, val_loader, test_loader = (
        make_dataloaders(
            args.model,
            args.batch_size,
        )
    )

    model = make_model(
        args.model
    ).to(DEVICE)

    if not AMP and not args.smoke:
        raise RuntimeError(
            "CUDA is unavailable. Stopping before "
            "a very long CPU training run."
        )

    if args.model == "ehr":
        train_labels = [
            int(s["label"])
            for s in train_loader.dataset.items
        ]
    else:
        train_labels = (
            train_loader.dataset.rows["label"]
            .astype(int)
            .tolist()
        )

    positives = sum(train_labels)
    negatives = len(train_labels) - positives

    if positives == 0 or negatives == 0:
        raise ValueError(
            "Invalid training class distribution"
        )

    pos_weight = min(
        float(negatives) / positives,
        10.0,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            device=DEVICE,
        )
    )

    lr = args.lr

    if lr is None:
        lr = (
            3e-4
            if args.model == "ehr"
            else 2e-5
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=AMP,
    )

    prefix = (
        f"trcg_stage2_{args.model}_seed{args.seed}"
    )

    best_path = CKPT / f"{prefix}_best.pt"
    last_path = CKPT / f"{prefix}_last.pt"
    done_path = CKPT / f"{prefix}.done"

    history_path = (
        LOGS / f"{prefix}_history.csv"
    )

    prediction_path = (
        RESULTS / f"{prefix}_test_predictions.csv"
    )

    summary_path = (
        RESULTS / f"{prefix}_summary.json"
    )

    print(
        f"Model: {args.model} | "
        f"Device: {DEVICE} | "
        f"LR: {lr} | "
        f"pos_weight={pos_weight:.3f}",
        flush=True,
    )

    start_epoch = 1
    best_ap = -1.0
    best_epoch = 0
    bad = 0

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if args.resume and last_path.exists():
        state = torch.load(
            last_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            state["model"]
        )

        optimizer.load_state_dict(
            state["optimizer"]
        )

        scaler.load_state_dict(
            state["scaler"]
        )

        start_epoch = int(
            state["epoch"]
        ) + 1

        best_ap = float(
            state["best_ap"]
        )

        best_epoch = int(
            state["best_epoch"]
        )

        bad = int(
            state["bad"]
        )

        print(
            f"RESUME from completed epoch "
            f"{start_epoch - 1}; "
            f"best epoch {best_epoch}; "
            f"bad={bad}",
            flush=True,
        )

    # --------------------------------------------------------
    # SMOKE TEST
    # --------------------------------------------------------

    if args.smoke:
        model.train()

        x, y, _ = next(
            iter(train_loader)
        )

        x = to_device(x)
        y = y.to(DEVICE)

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            "cuda",
            enabled=AMP,
        ):
            logits = model(x)

        loss = criterion(
            logits.float(),
            y.float(),
        )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(
            "SMOKE TEST PASSED:",
            args.model,
            "loss=",
            float(loss.item()),
        )

        return

    # --------------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        if bad >= args.patience:
            print(
                "Early-stopping patience already reached. "
                "Skipping further training."
            )
            break

        model.train()

        bar = tqdm(
            train_loader,
            desc=f"{args.model.upper()} epoch {epoch}",
        )

        for x, y, _ in bar:
            x = to_device(x)
            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=AMP,
            ):
                logits = model(x)

            # BCEWithLogits outside autocast.
            loss = criterion(
                logits.float(),
                y.float(),
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}"
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val = evaluate(
            model,
            val_loader,
            criterion,
        )

        improved = (
            val["auprc"] > best_ap + 1e-6
        )

        if improved:
            best_ap = val["auprc"]
            best_epoch = epoch
            bad = 0

            atomic_save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_auprc": val["auprc"],
                    "val_auroc": val["auroc"],
                },
                best_path,
            )

        else:
            bad += 1

        print(
            f"Epoch {epoch}: "
            f"val loss={val['loss']:.4f} | "
            f"AUROC={val['auroc']:.4f} | "
            f"AUPRC={val['auprc']:.4f} | "
            f"best_epoch={best_epoch} | "
            f"patience={bad}/{args.patience}",
            flush=True,
        )

        # ----------------------------------------------------
        # RESUMABLE CHECKPOINT
        # ----------------------------------------------------

        atomic_save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_ap": best_ap,
                "best_epoch": best_epoch,
                "bad": bad,
            },
            last_path,
        )

        # ----------------------------------------------------
        # HISTORY CSV
        # ----------------------------------------------------

        row = {
            "epoch": epoch,
            "val_loss": val["loss"],
            "val_auroc": val["auroc"],
            "val_auprc": val["auprc"],
            "best_epoch": best_epoch,
        }

        with open(
            history_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(row),
            )

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(row)

        if bad >= args.patience:
            print("Early stopping.")
            break

    # ========================================================
    # FINAL TEST USING BEST VALIDATION CHECKPOINT
    # ========================================================

    if not best_path.exists():
        raise RuntimeError(
            f"No best checkpoint: {best_path}"
        )

    best = torch.load(
        best_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        best["model"]
    )

    test = evaluate(
        model,
        test_loader,
        criterion,
    )

    pd.DataFrame(
        {
            "stay_id": test["stay_ids"],
            "label": test["labels"],
            "probability": test["probabilities"],
        }
    ).to_csv(
        prediction_path,
        index=False,
    )

    summary = {
        "stage": 2,
        "model": args.model,
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "val_auroc": best["val_auroc"],
        "val_auprc": best["val_auprc"],
        "test_auroc": test["auroc"],
        "test_auprc": test["auprc"],
        "best_checkpoint": str(best_path),
        "prediction_file": str(prediction_path),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Only mark complete AFTER predictions and summary exist.
    done_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 64)

    print(
        f"{args.model.upper()} STAGE-2 TEST: "
        f"AUROC={test['auroc']:.4f} "
        f"AUPRC={test['auprc']:.4f}"
    )

    print(
        f"Saved summary: {summary_path}"
    )

    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["ehr", "cxr"],
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    run(
        parser.parse_args()
    )
