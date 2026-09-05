from pathlib import Path
import numpy as np
import csv
import json
import re

ROOT = Path("/home/featurize/work/USCT_repro")
OOF = ROOT / "USCT_download/condition_cache/inversionnet_oof5_b32_blocks2_e67"
OUT = ROOT / "paper_reproduction/cosign_usct"

DATA = OUT / "data"
MANIFEST = OUT / "manifests"

DATA.mkdir(parents=True, exist_ok=True)
MANIFEST.mkdir(parents=True, exist_ok=True)


def numeric_id(p: Path) -> int:
    m = re.search(r"(\d+)$", p.stem)
    if not m:
        raise ValueError(f"Cannot parse sample id: {p}")
    return int(m.group(1))


def find_train_files():
    candidates = []

    # Preferred expected location.
    train_dir = OOF / "train"
    if train_dir.exists():
        candidates = list(train_dir.glob("train_*.npz"))

    # Robust fallback.
    if not candidates:
        for p in OOF.rglob("train_*.npz"):
            s = str(p).lower()
            if "/checkpoints/" in s:
                continue
            candidates.append(p)

    candidates = sorted(set(candidates), key=numeric_id)
    return candidates


def find_test_files():
    test_dir = OOF / "test"
    files = sorted(test_dir.glob("test_*.npz"), key=numeric_id)
    return files


def load_record(path):
    with np.load(path, allow_pickle=False) as z:
        required = [
            "condition_speed",
            "condition_norm",
            "target_speed",
            "target_norm",
            "sample_index",
        ]

        for key in required:
            if key not in z:
                raise KeyError(f"{path}: missing key {key}")

        rec = {k: np.asarray(z[k]) for k in required}

    for key in ["condition_speed", "condition_norm",
                "target_speed", "target_norm"]:
        a = rec[key]

        if a.shape != (1, 256, 256):
            raise ValueError(
                f"{path}: {key} has shape {a.shape}, "
                "expected (1,256,256)"
            )

        if not np.isfinite(a).all():
            raise ValueError(f"{path}: non-finite values in {key}")

    return rec


train_files = find_train_files()
test_files = find_test_files()

print("============================================================")
print("C1 DATA DISCOVERY")
print("============================================================")
print("train OOF files:", len(train_files))
print("test files:", len(test_files))

if train_files:
    print("first train:", train_files[0])
    print("last train :", train_files[-1])

if test_files:
    print("first test :", test_files[0])
    print("last test  :", test_files[-1])

# Hard validation against the OOF audit.
if len(train_files) != 897:
    raise RuntimeError(
        f"Expected 897 OOF train files, found {len(train_files)}"
    )

if len(test_files) != 100:
    raise RuntimeError(
        f"Expected 100 test files, found {len(test_files)}"
    )

# Smoke:
# 8 OOF training samples.
# 4 samples from existing test1-4 only for pipeline validation.
# These are NOT used for formal model selection.
smoke_train = train_files[:8]
smoke_val = test_files[:4]


def build(files, split):
    targets_norm = []
    conds_norm = []
    targets_speed = []
    conds_speed = []
    rows = []

    for p in files:
        r = load_record(p)

        sid = int(np.asarray(r["sample_index"]).reshape(-1)[0])

        # CoSIGN load_npy expects [N,H,W], since it adds channel itself.
        targets_norm.append(r["target_norm"][0].astype(np.float32))
        conds_norm.append(r["condition_norm"][0].astype(np.float32))

        targets_speed.append(r["target_speed"][0].astype(np.float32))
        conds_speed.append(r["condition_speed"][0].astype(np.float32))

        rows.append({
            "split": split,
            "sample_index": sid,
            "source_file": str(p),
        })

    target_norm = np.stack(targets_norm)
    cond_norm = np.stack(conds_norm)
    target_speed = np.stack(targets_speed)
    cond_speed = np.stack(conds_speed)

    np.save(DATA / f"smoke_{split}_gt_norm.npy", target_norm)
    np.save(DATA / f"smoke_{split}_cond_norm.npy", cond_norm)

    # Speed arrays are useful for debugging, but small here.
    np.save(DATA / f"smoke_{split}_gt_speed.npy", target_speed)
    np.save(DATA / f"smoke_{split}_cond_speed.npy", cond_speed)

    with open(MANIFEST / f"smoke_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["split", "sample_index", "source_file"]
        )
        w.writeheader()
        w.writerows(rows)

    return {
        "num_samples": len(files),
        "target_norm_shape": list(target_norm.shape),
        "condition_norm_shape": list(cond_norm.shape),

        "target_norm_min": float(target_norm.min()),
        "target_norm_max": float(target_norm.max()),
        "target_norm_mean": float(target_norm.mean()),
        "target_norm_std": float(target_norm.std()),

        "condition_norm_min": float(cond_norm.min()),
        "condition_norm_max": float(cond_norm.max()),
        "condition_norm_mean": float(cond_norm.mean()),
        "condition_norm_std": float(cond_norm.std()),

        "target_speed_min": float(target_speed.min()),
        "target_speed_max": float(target_speed.max()),
        "target_speed_mean": float(target_speed.mean()),
        "target_speed_std": float(target_speed.std()),

        "condition_speed_min": float(cond_speed.min()),
        "condition_speed_max": float(cond_speed.max()),

        "nan_count": int(np.isnan(target_norm).sum()
                         + np.isnan(cond_norm).sum()),
        "inf_count": int(np.isinf(target_norm).sum()
                         + np.isinf(cond_norm).sum()),

        "sample_ids": [int(r["sample_index"]) for r in rows],
    }


audit = {
    "source": str(OOF),
    "purpose": "CoSIGN-USCT C1 smoke pipeline",
    "formal_training": False,
    "normalization": (
        "Reuse target_norm/condition_norm from frozen "
        "InversionNet OOF cache; do not renormalize."
    ),
    "train_source_policy": "OOF conditions only",
    "smoke_validation_policy": (
        "test1-4 are pipeline-smoke only; "
        "not used for formal hyperparameter selection"
    ),
    "train": build(smoke_train, "train"),
    "val": build(smoke_val, "val"),
}

with open(DATA / "c1_smoke_audit.json", "w") as f:
    json.dump(audit, f, indent=2)

print()
print("============================================================")
print("C1 SMOKE DATA AUDIT")
print("============================================================")
print(json.dumps(audit, indent=2))

# Final safety gates.
assert audit["train"]["nan_count"] == 0
assert audit["train"]["inf_count"] == 0
assert audit["val"]["nan_count"] == 0
assert audit["val"]["inf_count"] == 0

assert audit["train"]["target_norm_min"] >= -1.05
assert audit["train"]["target_norm_max"] <=  1.05

print()
print("[PASS] C1 smoke dataset ready.")
