import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


CENTER = 1502.5
SCALE = 102.5
DATA_RANGE = 205.0


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--gt",
        required=True,
    )

    ap.add_argument(
        "--hint",
        required=True,
    )

    ap.add_argument(
        "--raw_pred",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_np = np.load(
        args.gt
    ).astype(np.float32)

    hint_np = np.load(
        args.hint
    ).astype(np.float32)

    raw_np = np.load(
        args.raw_pred
    ).astype(np.float32)

    if gt_np.shape != (
        400,
        256,
        256,
    ):
        raise RuntimeError(
            gt_np.shape
        )

    if hint_np.shape != gt_np.shape:
        raise RuntimeError(
            "hint shape mismatch"
        )

    if raw_np.shape != gt_np.shape:
        raise RuntimeError(
            "raw prediction shape mismatch"
        )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device
    )

    gt = torch.from_numpy(
        gt_np
    ).to(device)

    hint = torch.from_numpy(
        hint_np
    ).to(device)

    raw = torch.from_numpy(
        raw_np
    ).to(device)

    # -----------------------------------------------
    # Raw C4 residual expressed directly in m/s.
    # -----------------------------------------------

    delta_mps = (
        raw - hint
    ) * SCALE

    # -----------------------------------------------
    # Baseline
    # -----------------------------------------------

    base_err = (
        hint - gt
    ) * SCALE

    base_mse_i = (
        base_err.square()
        .mean(dim=(1, 2))
    )

    base_mae_i = (
        base_err.abs()
        .mean(dim=(1, 2))
    )

    base_psnr_i = (
        10.0
        * torch.log10(
            DATA_RANGE ** 2
            / base_mse_i.clamp_min(
                1e-12
            )
        )
    )

    baseline = {
        "mse_mean":
            float(
                base_mse_i.mean()
                .item()
            ),

        "mae_mean":
            float(
                base_mae_i.mean()
                .item()
            ),

        "psnr_mean":
            float(
                base_psnr_i.mean()
                .item()
            ),
    }

    print()
    print("=" * 100)
    print(
        "INVERSIONNET BASELINE"
    )
    print("=" * 100)

    print(
        json.dumps(
            baseline,
            indent=2,
        )
    )

    print()
    print("=" * 100)
    print(
        "RAW RESIDUAL STATISTICS"
    )
    print("=" * 100)

    abs_delta = (
        delta_mps.abs()
    )

    print(
        "mean |delta| [m/s] =",
        float(
            abs_delta.mean().item()
        )
    )

    # torch.quantile on CUDA cannot handle this
    # ~26M-element tensor reliably. Quantiles are only
    # diagnostic statistics, so compute them on CPU.
    abs_delta_cpu = (
        abs_delta
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    for q in [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]:

        print(
            f"q{int(q*100):02d} "
            f"|delta| [m/s] =",
            float(
                np.quantile(
                    abs_delta_cpu,
                    q,
                )
            )
        )

    del abs_delta_cpu

    # -----------------------------------------------
    # Screening grid.
    #
    # Deliberately compact first pass.
    # -----------------------------------------------

    alphas = [
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
    ]

    taus_mps = [
        0.0,
        0.5,
        1.0,
        2.0,
        3.0,
    ]

    caps_mps = [
        5.0,
        10.0,
        20.0,
        None,
    ]

    modes = [
        "hard",
        "soft",
    ]

    rows = []

    best_pred = None
    best_row = None

    for mode in modes:

        for tau in taus_mps:

            abs_d = (
                delta_mps.abs()
            )

            sign_d = (
                delta_mps.sign()
            )

            if mode == "hard":

                gated = torch.where(
                    abs_d >= tau,
                    delta_mps,
                    torch.zeros_like(
                        delta_mps
                    ),
                )

            elif mode == "soft":

                gated = (
                    sign_d
                    * torch.relu(
                        abs_d - tau
                    )
                )

            else:
                raise ValueError(
                    mode
                )

            for cap in caps_mps:

                if cap is None:

                    delta_used = (
                        gated
                    )

                    cap_label = (
                        "inf"
                    )

                else:

                    delta_used = (
                        gated.clamp(
                            min=-cap,
                            max=cap,
                        )
                    )

                    cap_label = str(
                        cap
                    )

                for alpha in alphas:

                    # Convert m/s correction
                    # back into normalized space.
                    refined = (
                        hint
                        + alpha
                        * delta_used
                        / SCALE
                    )

                    refined = (
                        refined.clamp(
                            -1.0,
                            1.0,
                        )
                    )

                    err = (
                        refined - gt
                    ) * SCALE

                    mse_i = (
                        err.square()
                        .mean(
                            dim=(1, 2)
                        )
                    )

                    mae_i = (
                        err.abs()
                        .mean(
                            dim=(1, 2)
                        )
                    )

                    psnr_i = (
                        10.0
                        * torch.log10(
                            DATA_RANGE ** 2
                            / mse_i.clamp_min(
                                1e-12
                            )
                        )
                    )

                    mse_mean = float(
                        mse_i.mean()
                        .item()
                    )

                    mae_mean = float(
                        mae_i.mean()
                        .item()
                    )

                    psnr_mean = float(
                        psnr_i.mean()
                        .item()
                    )

                    mse_wins = int(
                        (
                            mse_i
                            < base_mse_i
                        )
                        .sum()
                        .item()
                    )

                    mae_wins = int(
                        (
                            mae_i
                            < base_mae_i
                        )
                        .sum()
                        .item()
                    )

                    gain = (
                        baseline[
                            "mse_mean"
                        ]
                        - mse_mean
                    )

                    row = {
                        "mode":
                            mode,

                        "tau_mps":
                            tau,

                        "cap_mps":
                            cap_label,

                        "alpha":
                            alpha,

                        "mse_mean":
                            mse_mean,

                        "mae_mean":
                            mae_mean,

                        "psnr_mean":
                            psnr_mean,

                        "mse_wins":
                            mse_wins,

                        "mae_wins":
                            mae_wins,

                        "mse_gain":
                            gain,

                        "mse_gain_pct":
                            (
                                gain
                                /
                                baseline[
                                    "mse_mean"
                                ]
                                * 100.0
                            ),
                    }

                    rows.append(
                        row
                    )

                    if (
                        best_row is None
                        or (
                            mse_mean,
                            mae_mean,
                        )
                        <
                        (
                            best_row[
                                "mse_mean"
                            ],
                            best_row[
                                "mae_mean"
                            ],
                        )
                    ):

                        best_row = dict(
                            row
                        )

                        best_pred = (
                            refined
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(
                                np.float32
                            )
                        )

    # -----------------------------------------------
    # Ranking by primary MSE.
    # -----------------------------------------------

    ranking = sorted(
        rows,
        key=lambda x: (
            x["mse_mean"],
            x["mae_mean"],
        )
    )

    for i, r in enumerate(
        ranking,
        start=1,
    ):
        r["rank"] = i

    print()
    print("=" * 145)
    print(
        "AGR VAL400 MSE RANKING"
    )
    print("=" * 145)

    print(
        f"{'rank':>4} "
        f"{'mode':>6} "
        f"{'tau':>7} "
        f"{'cap':>7} "
        f"{'alpha':>7} "
        f"{'MSE':>10} "
        f"{'MAE':>9} "
        f"{'PSNR':>9} "
        f"{'MSEwin':>9} "
        f"{'MAEwin':>9} "
        f"{'gain':>10} "
        f"{'gain%':>8}"
    )

    for r in ranking[:30]:

        print(
            f"{r['rank']:4d} "
            f"{r['mode']:>6s} "
            f"{r['tau_mps']:7.2f} "
            f"{r['cap_mps']:>7s} "
            f"{r['alpha']:7.3f} "
            f"{r['mse_mean']:10.4f} "
            f"{r['mae_mean']:9.4f} "
            f"{r['psnr_mean']:9.4f} "
            f"{r['mse_wins']:5d}/400 "
            f"{r['mae_wins']:5d}/400 "
            f"{r['mse_gain']:+10.4f} "
            f"{r['mse_gain_pct']:+7.3f}%"
        )

    # -----------------------------------------------
    # MAE-safe subset:
    #
    # must beat baseline MSE AND not worsen mean MAE.
    # -----------------------------------------------

    mae_safe = [
        r
        for r in ranking
        if (
            r["mse_mean"]
            <
            baseline["mse_mean"]
            and
            r["mae_mean"]
            <=
            baseline["mae_mean"]
        )
    ]

    print()
    print("=" * 100)
    print("MAE-SAFE RESULT")
    print("=" * 100)

    if mae_safe:

        best_mae_safe = min(
            mae_safe,
            key=lambda x: (
                x["mse_mean"],
                x["mae_mean"],
            )
        )

        print(
            json.dumps(
                best_mae_safe,
                indent=2,
            )
        )

    else:

        best_mae_safe = None

        print(
            "No configuration "
            "simultaneously improves "
            "mean MSE and mean MAE."
        )

    # -----------------------------------------------
    # Stable subset:
    # MSE improvement on >= 300/400 samples.
    # -----------------------------------------------

    stable = [
        r
        for r in ranking
        if r[
            "mse_wins"
        ] >= 300
    ]

    print()
    print("=" * 100)
    print("STABLE >=300/400 RESULT")
    print("=" * 100)

    if stable:

        best_stable = min(
            stable,
            key=lambda x: (
                x["mse_mean"],
                x["mae_mean"],
            )
        )

        print(
            json.dumps(
                best_stable,
                indent=2,
            )
        )

    else:

        best_stable = None

        print(
            "No >=300/400 "
            "configuration found."
        )

    print()
    print("=" * 100)
    print("BEST MSE RESULT")
    print("=" * 100)

    print(
        json.dumps(
            best_row,
            indent=2,
        )
    )

    # -----------------------------------------------
    # Save
    # -----------------------------------------------

    np.save(
        out /
        "best_agr_pred_norm.npy",
        best_pred,
    )

    with open(
        out /
        "summary.json",
        "w",
    ) as f:

        json.dump(
            {
                "protocol": {
                    "dataset":
                        "formal VAL400",

                    "raw_prediction":
                        "HP-C4 joint best "
                        "step3500, "
                        "sigma=11.71851402",

                    "test_used":
                        False,

                    "gate_units":
                        "m/s",

                    "primary_metric":
                        "mean MSE",
                },

                "baseline":
                    baseline,

                "best_mse":
                    best_row,

                "best_mae_safe":
                    best_mae_safe,

                "best_stable":
                    best_stable,

                "ranking":
                    ranking,
            },
            f,
            indent=2,
        )

    with open(
        out /
        "summary.csv",
        "w",
        newline="",
    ) as f:

        fields = [
            "rank",
            "mode",
            "tau_mps",
            "cap_mps",
            "alpha",
            "mse_mean",
            "mae_mean",
            "psnr_mean",
            "mse_wins",
            "mae_wins",
            "mse_gain",
            "mse_gain_pct",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            ranking
        )

    print()
    print(
        "[PASS] AGR VAL400 "
        "screening complete."
    )


if __name__ == "__main__":
    main()
