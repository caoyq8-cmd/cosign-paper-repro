import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


CENTER = 1502.5
SCALE = 102.5

EXPECTED_ORIGINAL_N = 900
EXPECTED_VALID_N = 897
KNOWN_MISSING_IDS = {320, 378, 810}


def speed_to_norm(x):
    return (x - CENTER) / SCALE


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--condition_root",
        default=(
            "/home/featurize/work/USCT_repro/USCT_download/"
            "condition_cache/inversionnet_oof5_b32_blocks2_e67"
        ),
    )

    ap.add_argument(
        "--data_dir",
        default=(
            "/home/featurize/work/USCT_repro/"
            "paper_reproduction/cosign_usct/data"
        ),
    )

    ap.add_argument(
        "--output",
        default=(
            "/home/featurize/work/USCT_repro/"
            "paper_reproduction/cosign_usct/data/"
            "c4_pair_manifest.json"
        ),
    )

    args = ap.parse_args()

    condition_root = Path(args.condition_root)
    data_dir = Path(args.data_dir)
    out_path = Path(args.output)

    train_files = sorted(
        (condition_root / "train").glob("train_*.npz")
    )

    print("num train npz =", len(train_files))

    if len(train_files) != EXPECTED_VALID_N:
        raise RuntimeError(
            f"Expected {EXPECTED_VALID_N} train npz files, "
            f"got {len(train_files)}"
        )

    # --------------------------------------------------------
    # Pass 1: collect true original sample IDs.
    # --------------------------------------------------------

    sample_records = []

    for p in train_files:
        z = np.load(p)

        if "sample_index" not in z.files:
            raise RuntimeError(
                f"{p}: missing sample_index"
            )

        sample_id = int(
            np.asarray(
                z["sample_index"]
            ).reshape(-1)[0]
        )

        # filename audit
        try:
            filename_id = int(
                p.stem.split("_")[-1]
            )
        except Exception:
            raise RuntimeError(
                f"Cannot parse sample id from {p.name}"
            )

        if filename_id != sample_id:
            raise RuntimeError(
                f"Filename/sample_index mismatch: "
                f"{p.name} vs {sample_id}"
            )

        sample_records.append(
            (sample_id, p)
        )

    ids = [x[0] for x in sample_records]

    print("\n===== ORIGINAL-ID AUDIT =====")
    print("n IDs      =", len(ids))
    print("n unique   =", len(set(ids)))
    print("min ID     =", min(ids))
    print("max ID     =", max(ids))

    if len(set(ids)) != EXPECTED_VALID_N:
        raise RuntimeError(
            "Duplicate sample IDs found"
        )

    expected_all = set(
        range(
            1,
            EXPECTED_ORIGINAL_N + 1
        )
    )

    actual_ids = set(ids)

    missing_ids = sorted(
        expected_all - actual_ids
    )

    extra_ids = sorted(
        actual_ids - expected_all
    )

    print("missing IDs =", missing_ids)
    print("extra IDs   =", extra_ids)

    if set(missing_ids) != KNOWN_MISSING_IDS:
        raise RuntimeError(
            f"Unexpected missing IDs: {missing_ids}"
        )

    if extra_ids:
        raise RuntimeError(
            f"Unexpected extra IDs: {extra_ids}"
        )

    # Packed npy row order is the sorted valid-ID order.
    valid_ids = sorted(actual_ids)

    id_to_row = {
        sample_id: row
        for row, sample_id
        in enumerate(valid_ids)
    }

    print("\n===== PACKED ROW MAPPING =====")

    check_ids = [
        1,
        319,
        321,
        377,
        379,
        809,
        811,
        897,
        898,
        899,
        900,
    ]

    for sample_id in check_ids:
        if sample_id in id_to_row:
            print(
                f"sample_id={sample_id:3d} "
                f"-> packed_row={id_to_row[sample_id]:3d}"
            )

    # --------------------------------------------------------
    # Load existing packed arrays.
    # --------------------------------------------------------

    full_hint_path = (
        data_dir /
        "full_train_cond_norm.npy"
    )

    full_gt_path = (
        data_dir /
        "full_train_gt_norm.npy"
    )

    hint_full = np.load(
        full_hint_path,
        mmap_mode="r",
    )

    gt_full = np.load(
        full_gt_path,
        mmap_mode="r",
    )

    print("\n===== PACKED ARRAYS =====")
    print(
        "full hint shape =",
        hint_full.shape,
    )
    print(
        "full gt shape   =",
        gt_full.shape,
    )

    expected_shape = (
        EXPECTED_VALID_N,
        256,
        256,
    )

    if hint_full.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected hint shape: "
            f"{hint_full.shape}"
        )

    if gt_full.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected GT shape: "
            f"{gt_full.shape}"
        )

    # --------------------------------------------------------
    # Pass 2: full 897-sample alignment audit.
    # --------------------------------------------------------

    folds = []

    max_hint_array_diff = 0.0
    max_gt_array_diff = 0.0

    max_hint_norm_formula_diff = 0.0
    max_gt_norm_formula_diff = 0.0

    mse_list = []
    mae_list = []

    worst_hint = None
    worst_gt = None

    # Iterate by ORIGINAL ID, not lexicographic filename.
    records_sorted = sorted(
        sample_records,
        key=lambda x: x[0],
    )

    for k, (sample_id, p) in enumerate(
        records_sorted
    ):
        z = np.load(p)

        required = {
            "condition_speed",
            "condition_norm",
            "target_speed",
            "target_norm",
            "sample_index",
            "oof_fold",
        }

        missing_keys = (
            required - set(z.files)
        )

        if missing_keys:
            raise RuntimeError(
                f"{p}: missing keys "
                f"{missing_keys}"
            )

        fold = int(
            np.asarray(
                z["oof_fold"]
            ).reshape(-1)[0]
        )

        folds.append(fold)

        cond_speed = np.asarray(
            z["condition_speed"],
            dtype=np.float32,
        )

        cond_norm = np.asarray(
            z["condition_norm"],
            dtype=np.float32,
        )

        target_speed = np.asarray(
            z["target_speed"],
            dtype=np.float32,
        )

        target_norm = np.asarray(
            z["target_norm"],
            dtype=np.float32,
        )

        if cond_norm.shape != (
            1,
            256,
            256,
        ):
            raise RuntimeError(
                f"{p}: bad condition_norm "
                f"shape={cond_norm.shape}"
            )

        if target_norm.shape != (
            1,
            256,
            256,
        ):
            raise RuntimeError(
                f"{p}: bad target_norm "
                f"shape={target_norm.shape}"
            )

        # ----------------------------------------------
        # Verify normalization.
        # ----------------------------------------------

        hint_formula_diff = float(
            np.max(
                np.abs(
                    cond_norm -
                    speed_to_norm(cond_speed)
                )
            )
        )

        gt_formula_diff = float(
            np.max(
                np.abs(
                    target_norm -
                    speed_to_norm(target_speed)
                )
            )
        )

        max_hint_norm_formula_diff = max(
            max_hint_norm_formula_diff,
            hint_formula_diff,
        )

        max_gt_norm_formula_diff = max(
            max_gt_norm_formula_diff,
            gt_formula_diff,
        )

        # ----------------------------------------------
        # Correct original-ID -> packed-row mapping.
        # ----------------------------------------------

        packed_row = id_to_row[
            sample_id
        ]

        hint_diff = float(
            np.max(
                np.abs(
                    hint_full[packed_row]
                    -
                    cond_norm[0]
                )
            )
        )

        gt_diff = float(
            np.max(
                np.abs(
                    gt_full[packed_row]
                    -
                    target_norm[0]
                )
            )
        )

        if (
            hint_diff >
            max_hint_array_diff
        ):
            max_hint_array_diff = (
                hint_diff
            )
            worst_hint = (
                sample_id,
                packed_row,
                str(p),
            )

        if gt_diff > max_gt_array_diff:
            max_gt_array_diff = gt_diff
            worst_gt = (
                sample_id,
                packed_row,
                str(p),
            )

        # ----------------------------------------------
        # Original OOF baseline in physical units.
        # ----------------------------------------------

        err = (
            cond_speed.astype(
                np.float64
            )
            -
            target_speed.astype(
                np.float64
            )
        )

        mse_list.append(
            float(
                np.mean(
                    err ** 2
                )
            )
        )

        mae_list.append(
            float(
                np.mean(
                    np.abs(err)
                )
            )
        )

        if (k + 1) % 100 == 0:
            print(
                f"checked "
                f"{k + 1}/"
                f"{EXPECTED_VALID_N}"
            )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    fold_counts = dict(
        sorted(
            Counter(folds).items()
        )
    )

    audit_path = (
        condition_root /
        "oof_audit.json"
    )

    with open(
        audit_path,
        "r",
        encoding="utf-8",
    ) as f:
        original_audit = json.load(f)

    mse_mean = float(
        np.mean(mse_list)
    )

    mae_mean = float(
        np.mean(mae_list)
    )

    print("\n" + "=" * 80)
    print("C4 PAIRING AUDIT RESULT")
    print("=" * 80)

    print(
        "valid IDs        =",
        len(valid_ids),
    )
    print(
        "missing IDs      =",
        missing_ids,
    )
    print(
        "fold counts      =",
        fold_counts,
    )

    print(
        "baseline MSE     =",
        mse_mean,
    )
    print(
        "baseline MAE     =",
        mae_mean,
    )

    print(
        "max hint packed-array diff =",
        max_hint_array_diff,
    )

    print(
        "worst hint mapping         =",
        worst_hint,
    )

    print(
        "max gt packed-array diff   =",
        max_gt_array_diff,
    )

    print(
        "worst gt mapping           =",
        worst_gt,
    )

    print(
        "max hint norm formula diff =",
        max_hint_norm_formula_diff,
    )

    print(
        "max gt norm formula diff   =",
        max_gt_norm_formula_diff,
    )

    # --------------------------------------------------------
    # Hard checks.
    # --------------------------------------------------------

    if max_hint_array_diff > 1e-6:
        raise RuntimeError(
            "full_train_cond_norm.npy "
            "does not align with "
            "OOF cache under valid-ID order"
        )

    if max_gt_array_diff > 1e-6:
        raise RuntimeError(
            "full_train_gt_norm.npy "
            "does not align with "
            "OOF cache under valid-ID order"
        )

    if abs(
        mse_mean -
        original_audit[
            "oof_mse_mean"
        ]
    ) > 1e-3:
        raise RuntimeError(
            "Recomputed OOF MSE "
            "does not match "
            "oof_audit.json"
        )

    if abs(
        mae_mean -
        original_audit[
            "oof_mae_mean"
        ]
    ) > 1e-4:
        raise RuntimeError(
            "Recomputed OOF MAE "
            "does not match "
            "oof_audit.json"
        )

    expected_fold_counts = {
        0: 180,
        1: 180,
        2: 179,
        3: 179,
        4: 179,
    }

    if fold_counts != (
        expected_fold_counts
    ):
        raise RuntimeError(
            f"Unexpected fold counts: "
            f"{fold_counts}"
        )

    # --------------------------------------------------------
    # Manifest.
    # --------------------------------------------------------

    manifest = {
        "status": "PASS",

        "normalization": {
            "formula": (
                "(speed_mps - 1502.5) "
                "/ 102.5"
            ),
            "center": CENTER,
            "scale": SCALE,
            "nominal_speed_range": [
                1400.0,
                1605.0,
            ],
        },

        "sample_id_convention": {
            "original_id_range": [
                1,
                900,
            ],
            "num_original": 900,
            "num_valid": 897,
            "missing_ids": missing_ids,
            "packed_array_order": (
                "ascending valid original "
                "sample IDs"
            ),
        },

        "train": {
            "n": EXPECTED_VALID_N,
            "gt_norm": str(
                full_gt_path
            ),
            "hint_oof_norm": str(
                full_hint_path
            ),
            "shape": [
                EXPECTED_VALID_N,
                256,
                256,
            ],
            "fold_counts": {
                str(k): int(v)
                for k, v
                in fold_counts.items()
            },
        },

        "smoke_train": {
            "gt_norm": str(
                data_dir /
                "smoke_train_gt_norm.npy"
            ),
            "hint_norm": str(
                data_dir /
                "smoke_train_cond_norm.npy"
            ),
            "n": 8,
        },

        "smoke_val": {
            "gt_norm": str(
                data_dir /
                "smoke_val_gt_norm.npy"
            ),
            "hint_norm": str(
                data_dir /
                "smoke_val_cond_norm.npy"
            ),
            "n": 4,
        },

        "val20": {
            "gt_norm": str(
                data_dir /
                "val20_gt_norm.npy"
            ),
            "hint_norm": str(
                data_dir /
                "val20_cond_norm.npy"
            ),
            "n": 20,
        },

        "oof_baseline": {
            "mse_mean_speed":
                mse_mean,
            "mae_mean_speed":
                mae_mean,
        },

        "alignment": {
            "max_hint_array_diff":
                max_hint_array_diff,
            "max_gt_array_diff":
                max_gt_array_diff,
            "max_hint_norm_formula_diff":
                max_hint_norm_formula_diff,
            "max_gt_norm_formula_diff":
                max_gt_norm_formula_diff,
        },

        "source_oof_audit":
            str(audit_path),
    }

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print(
        "[PASS] C4 paired-data "
        "preparation complete."
    )
    print(
        "manifest =",
        out_path,
    )


if __name__ == "__main__":
    main()
