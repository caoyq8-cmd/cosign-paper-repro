import argparse
import json
from pathlib import Path
import numpy as np

CENTER = 1502.5
SCALE = 102.5

def inspect_array(name, path, expected_n):
    x = np.load(path, mmap_mode="r")
    print("\n" + "=" * 80)
    print(name)
    print("path  =", path)
    print("shape =", x.shape)
    print("dtype =", x.dtype)
    if x.shape != (expected_n, 256, 256):
        raise RuntimeError(
            f"{name}: expected {(expected_n,256,256)}, got {x.shape}"
        )
    mn = float("inf")
    mx = float("-inf")
    total = 0.0
    total_sq = 0.0
    count = 0
    for i in range(0, len(x), 32):
        y = np.asarray(x[i:i+32], dtype=np.float32)
        if not np.isfinite(y).all():
            raise RuntimeError(f"{name}: NaN/Inf found")
        mn = min(mn, float(y.min()))
        mx = max(mx, float(y.max()))
        yd = y.astype(np.float64)
        total += float(yd.sum())
        total_sq += float((yd * yd).sum())
        count += yd.size
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    std = var ** 0.5
    print("range =", mn, mx)
    print("mean  =", mean)
    print("std   =", std)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="/home/featurize/work/USCT_repro/"
                "paper_reproduction/cosign_usct/data/"
                "c4_pair_manifest.json",
    )
    args = ap.parse_args()
    with open(args.manifest, "r", encoding="utf-8") as f:
        m = json.load(f)
    if m.get("status") != "PASS":
        raise RuntimeError("Manifest status is not PASS")

    gt = inspect_array(
        "TRAIN GT",
        m["train"]["gt_norm"],
        m["train"]["n"],
    )
    hint = inspect_array(
        "TRAIN OOF HINT",
        m["train"]["hint_oof_norm"],
        m["train"]["n"],
    )
    inspect_array(
        "SMOKE TRAIN GT",
        m["smoke_train"]["gt_norm"],
        m["smoke_train"]["n"],
    )
    inspect_array(
        "SMOKE TRAIN HINT",
        m["smoke_train"]["hint_norm"],
        m["smoke_train"]["n"],
    )
    inspect_array(
        "VAL20 GT",
        m["val20"]["gt_norm"],
        m["val20"]["n"],
    )
    inspect_array(
        "VAL20 HINT",
        m["val20"]["hint_norm"],
        m["val20"]["n"],
    )

    # Recompute full OOF baseline from normalized arrays.
    mse = []
    mae = []
    for i in range(0, len(gt), 16):
        g = np.asarray(gt[i:i+16], dtype=np.float64) * SCALE + CENTER
        h = np.asarray(hint[i:i+16], dtype=np.float64) * SCALE + CENTER
        e = h - g
        axes = tuple(range(1, e.ndim))
        mse.extend(np.mean(e**2, axis=axes).tolist())
        mae.extend(np.mean(np.abs(e), axis=axes).tolist())

    mse_mean = float(np.mean(mse))
    mae_mean = float(np.mean(mae))
    print("\n" + "=" * 80)
    print("OOF BASELINE RECOMPUTATION")
    print("MSE =", mse_mean)
    print("MAE =", mae_mean)

    ref_mse = m["oof_baseline"]["mse_mean_speed"]
    ref_mae = m["oof_baseline"]["mae_mean_speed"]
    print("reference MSE =", ref_mse)
    print("reference MAE =", ref_mae)

    if abs(mse_mean - ref_mse) > 1e-2:
        raise RuntimeError("MSE audit failed")
    if abs(mae_mean - ref_mae) > 1e-3:
        raise RuntimeError("MAE audit failed")

    print("\n[PASS] C4 paired dataset audit complete.")

if __name__ == "__main__":
    main()
