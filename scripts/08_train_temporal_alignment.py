"""TRCG Stage 3: frozen unimodal features, time-correct EHR/CXR alignment.

Run inside C:\\MedModWork\\TRCGFusion with existing Stage 2 checkpoints.

python scripts/08_train_temporal_alignment.py cache
python scripts/08_train_temporal_alignment.py train --resume
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.trcg_encoders import EHRMortality, CXRMortality

CACHE = ROOT / "cache"
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
for directory in (CACHE, CKPT, RESULTS, LOGS):
    directory.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"

EHR_PT = CKPT / "trcg_stage2_ehr_seed42_best.pt"
CXR_PT = CKPT / "trcg_stage2_cxr_seed42_best.pt"
MANIFEST = CACHE / "trcg_manifest.csv"
PREPROCESSOR = CACHE / "ehr_preprocessor.json"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if AMP:
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload, path):
    temp = path.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def save_npz_atomic(path, **data):
    temp = path.with_suffix(".npz.tmp")
    with open(temp, "wb") as f:
        np.savez_compressed(f, **data)
    os.replace(temp, path)


class PairedStage3(Dataset):
    def __init__(self, split):
        rows = pd.read_csv(MANIFEST)
        self.rows = rows.loc[rows["split"] == split].to_dict("records")
        events = torch.load(
            CACHE / f"{split}_event_cache.pt",
            map_location="cpu",
            weights_only=False,  # trusted files created by our own preprocessing
        )
        self.event_by_stay = {str(s["stay_id"]): s for s in events}
        assert len(self.event_by_stay) == len(events), "Duplicate event-cache stay IDs"
        assert self.rows, f"No {split} records"
        missing = [str(r["stay_id"]) for r in self.rows
                   if str(r["stay_id"]) not in self.event_by_stay]
        assert not missing, f"Missing cached EHR events: {missing[:5]}"

        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        events = self.event_by_stay[str(r["stay_id"])]
        with Image.open(r["cxr_path"]) as im:
            image = self.tf(im.convert("RGB"))
        cxr_hour = float(r["cxr_hour"])
        assert 0 <= cxr_hour <= 48
        return events, image, cxr_hour, int(r["label"]), str(r["stay_id"])


def collate(batch):
    events, images, hours, labels, ids = zip(*batch)
    lens = torch.tensor([len(s["times"]) for s in events], dtype=torch.long)
    if torch.any(lens <= 0):
        raise RuntimeError("A patient has no EHR timestamps")
    B, T, V = len(batch), int(lens.max()), 17
    ehr = {
        "values": torch.zeros(B, T, V),
        "categories": torch.zeros(B, T, V, dtype=torch.long),
        "mask": torch.zeros(B, T, V),
        "delta": torch.zeros(B, T, V),
        "times": torch.zeros(B, T),
        "lengths": lens,
    }
    for i, sample in enumerate(events):
        n = int(lens[i])
        for name in ("values", "categories", "mask", "delta"):
            ehr[name][i, :n] = sample[name]
        ehr["times"][i, :n] = sample["times"]
    return (ehr, torch.stack(images),
            torch.tensor(hours, dtype=torch.float32),
            np.asarray(labels, dtype=np.int64), list(ids))


def load_encoders():
    with PREPROCESSOR.open("r", encoding="utf-8") as f:
        prep = json.load(f)
    ehr = EHRMortality(prep["categorical_vocab"])
    cxr = CXRMortality(None)  # full Stage-2 checkpoint restores encoder weights
    for path, model in ((EHR_PT, ehr), (CXR_PT, cxr)):
        if not path.exists():
            raise FileNotFoundError(f"Stage-2 best checkpoint missing: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
    return ehr.to(DEVICE).eval(), cxr.to(DEVICE).eval()


@torch.inference_mode()
def extract_split(split, batch_size):
    ds = PairedStage3(split)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=AMP, collate_fn=collate,
    )
    ehr_model, cxr_model = load_encoders()
    accum = {key: [] for key in (
        "ehr_at_cxr", "ehr_at_48", "cxr", "cxr_hour", "cxr_age",
        "anchor_valid", "label", "stay_id",
    )}

    for ehr, image, cxr_hour, label, ids in tqdm(loader, desc=f"Caching {split}"):
        ehr = {k: v.to(DEVICE, non_blocking=True) for k, v in ehr.items()}
        image = image.to(DEVICE, non_blocking=True)
        cxr_hour = cxr_hour.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=AMP):
            final, sequence = ehr_model.encoder(ehr, return_sequence=True)
            cxr_vec = cxr_model.encoder(image)
        # Determine the most recent EHR timestamp no later than the CXR.
        # The encoder is causal, so this state has no post-CXR leakage.
        valid_time = (
            torch.arange(sequence.shape[1], device=DEVICE)[None, :]
            < ehr["lengths"][:, None]
        )
        before_cxr = valid_time & (ehr["times"] <= cxr_hour[:, None] + 1e-6)
        last_before = before_cxr.long().sum(dim=1) - 1
        valid = last_before >= 0
        anchor = sequence[
            torch.arange(len(ids), device=DEVICE),
            last_before.clamp(min=0),
        ].float()
        anchor[~valid] = 0.0  # no EHR available at image time
        append = {
            "ehr_at_cxr": anchor.cpu().numpy(),
            "ehr_at_48": final.float().cpu().numpy(),
            "cxr": cxr_vec.float().cpu().numpy(),
            "cxr_hour": cxr_hour.cpu().numpy(),
            "cxr_age": (48.0 - cxr_hour).cpu().numpy(),
            "anchor_valid": valid.cpu().numpy(),
            "label": label,
            "stay_id": np.asarray(ids, dtype=str),
        }
        for name, arr in append.items():
            accum[name].append(arr)

    result = {k: np.concatenate(v) for k, v in accum.items()}
    output = CACHE / f"trcg_stage3_{split}_features.npz"
    save_npz_atomic(output, **result)
    print(f"Saved: {output}")
    print(f"Split {split}: {len(result['stay_id'])} stays, "
          f"{int(result['anchor_valid'].sum())} with pre-CXR EHR")


class AlignmentProjectors(nn.Module):
    def __init__(self):
        super().__init__()
        self.ehr = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 128),
            nn.GELU(), nn.Linear(128, 128),
        )
        self.cxr = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 128),
            nn.GELU(), nn.Linear(128, 128),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, e, c):
        e = F.normalize(self.ehr(e), dim=1)
        c = F.normalize(self.cxr(c), dim=1)
        return e, c


def contrastive_loss(e, c, logit_scale):
    scale = logit_scale.exp().clamp(max=100)
    similarity = scale * (e @ c.T)
    target = torch.arange(len(e), device=e.device)
    return 0.5 * (
        F.cross_entropy(similarity, target)
        + F.cross_entropy(similarity.T, target)
    ), similarity


def load_features(split):
    path = CACHE / f"trcg_stage3_{split}_features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run cache first: {path}")
    with np.load(path) as z:
        valid = z["anchor_valid"].astype(bool)
        e = torch.from_numpy(z["ehr_at_cxr"][valid].astype("float32"))
        c = torch.from_numpy(z["cxr"][valid].astype("float32"))
        n_total = len(valid)
    if len(e) < 2:
        raise RuntimeError(f"Too few valid {split} EHR-CXR anchors")
    print(f"{split} alignment pairs: {len(e)}/{n_total}")
    return e, c


@torch.no_grad()
def evaluate_alignment(model, e, c):
    model.eval()
    a, b = model(e, c)
    loss, similarity = contrastive_loss(a, b, model.logit_scale)
    target = torch.arange(len(a), device=a.device)
    recall_at_1 = (
        0.5 * (similarity.argmax(1).eq(target).float().mean()
               + similarity.argmax(0).eq(target).float().mean())
    )
    return float(loss.item()), float(recall_at_1.item())


def train(args):
    seed_everything(args.seed)
    train_e, train_c = load_features("train")
    val_e, val_c = load_features("val")
    val_e, val_c = val_e.to(DEVICE), val_c.to(DEVICE)
    loader = DataLoader(
        TensorDataset(train_e, train_c),
        shuffle=True, batch_size=args.batch_size,
        drop_last=True, num_workers=0,
    )

    model = AlignmentProjectors().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    last = CKPT / f"trcg_stage3_align_seed{args.seed}_last.pt"
    best = CKPT / f"trcg_stage3_align_seed{args.seed}_best.pt"
    done = CKPT / f"trcg_stage3_align_seed{args.seed}.done"
    summary_path = RESULTS / f"trcg_stage3_align_seed{args.seed}_summary.json"
    history = LOGS / f"trcg_stage3_align_seed{args.seed}_history.csv"
    start, best_loss, best_epoch, bad = 1, float("inf"), 0, 0

    if args.resume and last.exists():
        state = torch.load(last, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["epoch"] + 1
        best_loss = state["best_loss"]
        best_epoch = state["best_epoch"]
        bad = state["bad"]
        print(f"Resumed after epoch {start - 1}. Best epoch {best_epoch}.")

    print(f"Stage 3: device={DEVICE}, batch_size={args.batch_size}, "
          f"train_pairs={len(train_e)}, val_pairs={len(val_e)}")
    for epoch in range(start, args.epochs + 1):
        if bad >= args.patience:
            print("Early-stopping patience already reached")
            break
        model.train()
        losses = []
        for e, c in loader:
            e, c = e.to(DEVICE), c.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            a, b = model(e, c)
            loss, _ = contrastive_loss(a, b, model.logit_scale)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite alignment loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_loss, recall = evaluate_alignment(model, val_e, val_c)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            bad = 0
            atomic_torch_save({
                "model": model.state_dict(),
                "best_epoch": best_epoch,
                "val_alignment_loss": val_loss,
                "val_recall_at_1": recall,
            }, best)
        else:
            bad += 1
        atomic_torch_save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss, "best_epoch": best_epoch, "bad": bad,
        }, last)
        with open(history, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["epoch", "train_loss", "val_loss", "val_recall_at_1"])
            writer.writerow([epoch, np.mean(losses), val_loss, recall])
        print(f"Epoch {epoch}: train={np.mean(losses):.4f} "
              f"val={val_loss:.4f} recall@1={recall:.4f} "
              f"best={best_epoch} patience={bad}/{args.patience}", flush=True)

    if not best.exists():
        raise RuntimeError("Best projection checkpoint absent")
    best_state = torch.load(best, map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_state["model"])
    final_loss, final_recall = evaluate_alignment(model, val_e, val_c)
    summary = {
        "stage": 3,
        "method": "frozen-encoder temporally anchored cross-modal alignment",
        "seed": args.seed,
        "train_alignment_pairs": int(len(train_e)),
        "val_alignment_pairs": int(len(val_e)),
        "best_epoch": int(best_state["best_epoch"]),
        "best_val_alignment_loss": float(final_loss),
        "best_val_recall_at_1": float(final_recall),
        "projection_checkpoint": str(best),
        "ehr_encoder_checkpoint": str(EHR_PT),
        "cxr_encoder_checkpoint": str(CXR_PT),
        "notes": "Mortality labels are not used in Stage-3 loss. Encoders frozen; projection heads trained with in-batch negatives. No test evaluation.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    done.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Stage 3 complete:", summary_path)



def smoke():
    """Check checkpoint loading, paired forward and contrastive gradients."""
    loader = DataLoader(
        PairedStage3("train"), batch_size=4, shuffle=False,
        collate_fn=collate, num_workers=0,
    )
    ehr, img, cxr_hour, _, ids = next(iter(loader))
    em, cm = load_encoders()
    ehr = {k: v.to(DEVICE) for k, v in ehr.items()}
    img, cxr_hour = img.to(DEVICE), cxr_hour.to(DEVICE)
    with torch.no_grad():
        _, sequence = em.encoder(ehr, return_sequence=True)
        cxr = cm.encoder(img)
        valid_time = torch.arange(sequence.shape[1], device=DEVICE)[None, :] < ehr["lengths"][:, None]
        last_before = (valid_time & (ehr["times"] <= cxr_hour[:, None] + 1e-6)).long().sum(1) - 1
        valid = last_before >= 0
        anchor = sequence[torch.arange(len(ids), device=DEVICE), last_before.clamp(min=0)].float()
    assert anchor.shape == (4, 128)
    assert cxr.shape == (4, 512)
    projectors = AlignmentProjectors().to(DEVICE)
    a, b = projectors(anchor, cxr.float())
    loss, _ = contrastive_loss(a, b, projectors.logit_scale)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"STAGE 3 SMOKE TEST PASSED: loss={loss.item():.4f}, "
          f"pre-CXR EHR available={int(valid.sum())}/4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "cache", "train"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "cache":
        for split in ("train", "val"):
            p = CACHE / f"trcg_stage3_{split}_features.npz"
            if p.exists():
                print(f"Skip existing cache {p}")
                continue
            extract_split(split, args.cache_batch_size)
    else:
        train(args)


if __name__ == "__main__":
    main()
