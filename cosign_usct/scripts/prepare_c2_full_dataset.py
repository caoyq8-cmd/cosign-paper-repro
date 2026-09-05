from pathlib import Path
import numpy as np
import json
import csv
import re

ROOT = Path("/home/featurize/work/USCT_repro")

CACHE = (
    ROOT
    / "USCT_download"
    / "condition_cache"
    / "inversionnet_oof5_b32_blocks2_e67"
)

OUT = (
    ROOT
    / "paper_reproduction"
    / "cosign_usct"
)

DATA = OUT / "data"
MAN = OUT / "manifests"

DATA.mkdir(parents=True, exist_ok=True)
MAN.mkdir(parents=True, exist_ok=True)


def sid(p):
    m = re.search(r"(\d+)$", p.stem)
    if not m:
        raise RuntimeError(f"cannot parse sample ID: {p}")
    return int(m.group(1))


def find_train():
    # Expected official OOF output directory.
    d = CACHE / "train"

    if d.exists():
        fs = sorted(d.glob("train_*.npz"), key=sid)
        if len(fs) == 897:
            return fs

    # Robust fallback.
    fs = []
    for p in CACHE.rglob("train_*.npz"):
        if "checkpoint" not in str(p).lower():
            fs.append(p)

    # Deduplicate by resolved path / sample ID.
    by_id = {}

    for p in fs:
        i = sid(p)
        if i not in by_id:
            by_id[i] = p

    return [by_id[i] for i in sorted(by_id)]


def find_test():
    return sorted(
        (CACHE / "test").glob("test_*.npz"),
        key=sid
    )


def load_one(p):
    with np.load(p, allow_pickle=False) as z:
        required = {
            "target_norm",
            "condition_norm",
            "target_speed",
            "condition_speed",
            "sample_index",
        }

        missing = required - set(z.keys())

        if missing:
            raise RuntimeError(
                f"{p} missing keys: {sorted(missing)}"
            )

        target = z["target_norm"].astype(np.float32)
        cond = z["condition_norm"].astype(np.float32)
        speed = z["target_speed"].astype(np.float32)

        sample_id = int(
            np.asarray(z["sample_index"]).reshape(-1)[0]
        )

    assert target.shape == (1, 256, 256)
    assert cond.shape == (1, 256, 256)
    assert speed.shape == (1, 256, 256)

    assert np.isfinite(target).all()
    assert np.isfinite(cond).all()
    assert np.isfinite(speed).all()

    return sample_id, target[0], cond[0], speed[0]


train_files = find_train()
test_files = find_test()

print("train files =", len(train_files))
print("test files  =", len(test_files))

assert len(train_files) == 897, len(train_files)
assert len(test_files) == 100, len(test_files)

# ---------------------------------------------------------
# Formal protocol
# ---------------------------------------------------------
# train: all 897 training samples
# val20: test IDs 1-20 (development/tuning set)
# DEV30 remains test IDs 21-50
# test51-100 is NOT touched here.
# ---------------------------------------------------------

val20_files = [
    p for p in test_files
    if 1 <= sid(p) <= 20
]

assert len(val20_files) == 20


def export(files, name):
    gt = []
    cond = []
    speed = []
    rows = []

    for p in files:
        i, x, c, s = load_one(p)

        gt.append(x)
        cond.append(c)
        speed.append(s)

        rows.append({
            "sample_id": i,
            "source": str(p),
        })

    gt = np.stack(gt).astype(np.float32)
    cond = np.stack(cond).astype(np.float32)
    speed = np.stack(speed).astype(np.float32)

    np.save(DATA / f"{name}_gt_norm.npy", gt)
    np.save(DATA / f"{name}_cond_norm.npy", cond)

    with open(MAN / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sample_id", "source"],
        )
        w.writeheader()
        w.writerows(rows)

    return {
        "n": int(len(files)),
        "gt_shape": list(gt.shape),
        "cond_shape": list(cond.shape),

        "gt_min": float(gt.min()),
        "gt_max": float(gt.max()),
        "gt_mean": float(gt.mean()),
        "gt_std": float(gt.std()),

        "cond_min": float(cond.min()),
        "cond_max": float(cond.max()),
        "cond_mean": float(cond.mean()),
        "cond_std": float(cond.std()),

        "speed_min": float(speed.min()),
        "speed_max": float(speed.max()),
        "speed_mean": float(speed.mean()),
        "speed_std": float(speed.std()),

        "nan": int(np.isnan(gt).sum() + np.isnan(cond).sum()),
        "inf": int(np.isinf(gt).sum() + np.isinf(cond).sum()),

        "sample_ids": [r["sample_id"] for r in rows],
    }


audit = {
    "protocol": {
        "train": "897 OOF training targets",
        "val20": "test 1-20, development/tuning only",
        "dev30": "test 21-50, not exported in C2",
        "holdout": "test 51-100 untouched in C2",
    },
    "normalization": (
        "reuse target_norm and condition_norm from "
        "OOF InversionNet cache; no second normalization"
    ),
    "train": export(train_files, "full_train"),
    "val20": export(val20_files, "val20"),
}

with open(DATA / "c2_dataset_audit.json", "w") as f:
    json.dump(audit, f, indent=2)

print(json.dumps(audit, indent=2))

assert audit["train"]["n"] == 897
assert audit["val20"]["n"] == 20

assert audit["train"]["nan"] == 0
assert audit["train"]["inf"] == 0

assert audit["train"]["gt_min"] >= -1.05
assert audit["train"]["gt_max"] <= 1.05

print()
print("[PASS] C2 full dataset ready.")
