r"""Read-only Stage 8 audit: user-owned MedMod CXR checkpoints and DenseNet source.
Run from C:\MedModWork\TRCGFusion with the MedMod venv active.
"""
from pathlib import Path
import json
import re
import torch

REPO = Path(r"C:\MedModWork\MedModRepo")
OUT = Path(r"C:\MedModWork\TRCGFusion\logs\stage8_checkpoint_audit.txt")
NAMES = [
    "checkpoints_local/densenet121_best.pt",
    "checkpoints_local/cxr_best.pt",
    "ssl456_checkpoints/convirt_cxr_finetune_seed42_best.pt",
    "ssl456_checkpoints/align_cxr_finetune_seed42_best.pt",
]
lines = []
def say(s=""):
    s = str(s)
    lines.append(s)
    print(s)

say("STAGE 8 READ-ONLY CXR CHECKPOINT AUDIT")
say(f"Torch {torch.__version__}; CUDA available={torch.cuda.is_available()}")

for rel in NAMES:
    p = REPO / Path(rel)
    say("\n" + "=" * 72)
    say(f"CHECKPOINT: {p}")
    if not p.is_file():
        say("MISSING")
        continue
    say(f"Size MB: {p.stat().st_size / 1024**2:.1f}")
    try:
        # These are locally produced, trusted checkpoints; no arbitrary downloads.
        try:
            obj = torch.load(p, map_location="cpu", weights_only=True)
        except Exception as exc:
            say(f"Safe weights-only load failed: {type(exc).__name__}: {str(exc)[:250]}")
            say("Not attempting unsafe pickle loading. Ask for further inspection.")
            continue
        say(f"Root type: {type(obj).__name__}")
        if not isinstance(obj, dict):
            say("Not a dict checkpoint; inspect the original model loader.")
            continue
        say(f"Root keys: {list(obj.keys())[:30]}")
        for k, v in obj.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                say(f"  metadata {k}: {str(v)[:180]}")
        possible = [("root", obj)]
        possible += [(str(k), v) for k, v in obj.items() if isinstance(v, dict)]
        possible.sort(key=lambda kv: sum(isinstance(x, torch.Tensor) for x in kv[1].values()), reverse=True)
        for name, sd in possible[:3]:
            tensors = [(k, v) for k, v in sd.items() if isinstance(v, torch.Tensor)]
            if not tensors:
                continue
            say(f"  STATE CANDIDATE [{name}]: {len(tensors)} tensors")
            say("  first 12: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in tensors[:12]))
            patterns = ["denseblock", "conv0.weight", "classifier", "head.weight", "fc.weight", "backbone", "encoder"]
            relevant = [(k, v) for k, v in tensors if any(pt in k.lower() for pt in patterns)]
            say("  architecture/head examples:")
            for k, v in (relevant[:6] + relevant[-8:]):
                say(f"    {k}: {tuple(v.shape)}")
    except Exception as exc:
        say(f"ERROR: {type(exc).__name__}: {exc}")

say("\n" + "=" * 72)
say("DENSENET DEFINITIONS AND TRANSFORMS IN EXISTING REPO")
pattern = re.compile(r"densenet121|densenet|DenseNet|class\s+.*CXR|ImageNet|Normalize\(|Resize\(", re.I)
matches = []
for py in REPO.rglob("*.py"):
    if any(part.lower() in {".venv", "venv", "__pycache__", ".git"} for part in py.parts):
        continue
    try:
        code = py.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        continue
    hits = [(i, line.strip()) for i, line in enumerate(code, 1) if pattern.search(line)]
    if hits:
        dense_hits = sum("densenet" in line.lower() for _, line in hits)
        matches.append((dense_hits, str(py).lower(), py, hits))

matches.sort(key=lambda x: (-x[0], x[1]))
for _, __, py, hits in matches[:35]:
    say(f"\nFILE: {py}")
    for i, line in hits[:12]:
        say(f"  L{i}: {line[:170]}")
    if len(hits) > 12:
        say(f"  ... plus {len(hits) - 12} additional matches")
if len(matches) > 35:
    say(f"... {len(matches)-35} additional matching source files not listed")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
say(f"\nREPORT SAVED: {OUT}")
