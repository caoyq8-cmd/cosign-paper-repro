import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCALE = 102.5
DATA_RANGE = 205.0


def gaussian_kernel(sigma, device):
    if sigma <= 0:
        return None

    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.float32,
        device=device,
    )

    k1 = torch.exp(
        -(x ** 2) / (2.0 * sigma ** 2)
    )
    k1 = k1 / k1.sum()

    k2 = k1[:, None] * k1[None, :]
    k2 = k2[None, None]

    return k2


def smooth_delta(delta, sigma, batch_size=20):
    if sigma <= 0:
        return delta.clone()

    device = delta.device
    kernel = gaussian_kernel(sigma, device)
    radius = kernel.shape[-1] // 2

    outs = []

    for start in range(0, len(delta), batch_size):
        end = min(start + batch_size, len(delta))

        x = delta[start:end, None]

        x = F.pad(
            x,
            (radius, radius, radius, radius),
            mode="reflect",
        )

        y = F.conv2d(
            x,
            kernel,
        )

        outs.append(y[:, 0])

    return torch.cat(outs, dim=0)


def metrics(pred, gt):
    err = (pred - gt) * SCALE

    mse_i = (
        err.square()
        .mean(dim=(1, 2))
    )

    mae_i = (
        err.abs()
        .mean(dim=(1, 2))
    )

    psnr_i = (
        10.0
        * torch.log10(
            DATA_RANGE ** 2
            / mse_i.clamp_min(1e-12)
        )
    )

    return mse_i, mae_i, psnr_i


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gt", required=True)
    ap.add_argument("--hint", required=True)
    ap.add_argument("--raw_pred", required=True)
    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt_np = np.load(args.gt).astype(np.float32)
    hint_np = np.load(args.hint).astype(np.float32)
    raw_np = np.load(args.raw_pred).astype(np.float32)

    if gt_np.shape != (400, 256, 256):
        raise RuntimeError(gt_np.shape)

    if hint_np.shape != gt_np.shape:
        raise RuntimeError("hint shape mismatch")

    if raw_np.shape != gt_np.shape:
        raise RuntimeError("raw prediction shape mismatch")

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device =", device)

    gt = torch.from_numpy(gt_np).to(device)
    hint = torch.from_numpy(hint_np).to(device)
    raw = torch.from_numpy(raw_np).to(device)

    # Raw generative correction in m/s.
    delta = (raw - hint) * SCALE

    base_mse, base_mae, base_psnr = metrics(
        hint,
        gt,
    )

    baseline = {
        "mse_mean": float(base_mse.mean()),
        "mae_mean": float(base_mae.mean()),
        "psnr_mean": float(base_psnr.mean()),
    }

    print()
    print("=" * 100)
    print("INVERSIONNET BASELINE")
    print("=" * 100)
    print(json.dumps(baseline, indent=2))

    # ------------------------------------------------
    # Screening region.
    # sigma_px=0 reproduces unsmoothed residual.
    # ------------------------------------------------

    smooth_sigmas = [
        0.0,
        0.75,
        1.5,
        3.0,
        5.0,
    ]

    taus = [
        0.0,
        0.5,
        1.0,
        2.0,
        3.0,
    ]

    alphas = [
        0.03,
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
    ]

    rows = []

    best = None
    best_pred = None

    for smooth_sigma in smooth_sigmas:

        print()
        print("=" * 100)
        print(
            "smooth_sigma_px =",
            smooth_sigma
        )
        print("=" * 100)

        delta_s = smooth_delta(
            delta,
            smooth_sigma,
            batch_size=20,
        )

        abs_d = delta_s.abs()
        sign_d = delta_s.sign()

        for tau in taus:

            # Soft threshold / shrinkage.
            delta_g = (
                sign_d
                * torch.relu(
                    abs_d - tau
                )
            )

            for alpha in alphas:

                refined = (
                    hint
                    + alpha
                    * delta_g
                    / SCALE
                )

                refined = refined.clamp(
                    -1.0,
                    1.0,
                )

                mse_i, mae_i, psnr_i = metrics(
                    refined,
                    gt,
                )

                mse_mean = float(
                    mse_i.mean()
                )

                mae_mean = float(
                    mae_i.mean()
                )

                psnr_mean = float(
                    psnr_i.mean()
                )

                mse_wins = int(
                    (mse_i < base_mse)
                    .sum()
                )

                mae_wins = int(
                    (mae_i < base_mae)
                    .sum()
                )

                gain = (
                    baseline["mse_mean"]
                    - mse_mean
                )

                row = {
                    "smooth_sigma_px":
                        smooth_sigma,

                    "tau_mps":
                        tau,

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
                        100.0
                        * gain
                        / baseline["mse_mean"],
                }

                rows.append(row)

                if (
                    best is None
                    or (
                        mse_mean,
                        mae_mean,
                    )
                    <
                    (
                        best["mse_mean"],
                        best["mae_mean"],
                    )
                ):
                    best = dict(row)

                    best_pred = (
                        refined.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

    ranking = sorted(
        rows,
        key=lambda x: (
            x["mse_mean"],
            x["mae_mean"],
        )
    )

    for rank, r in enumerate(
        ranking,
        start=1,
    ):
        r["rank"] = rank

    print()
    print("=" * 145)
    print("SRR VAL400 MSE RANKING")
    print("=" * 145)

    for r in ranking[:30]:
        print(
            f"#{r['rank']:02d} "
            f"smooth={r['smooth_sigma_px']:4.2f} "
            f"tau={r['tau_mps']:4.1f} "
            f"alpha={r['alpha']:5.3f} "
            f"MSE={r['mse_mean']:9.4f} "
            f"MAE={r['mae_mean']:7.4f} "
            f"PSNR={r['psnr_mean']:7.4f} "
            f"MSEwin={r['mse_wins']:3d}/400 "
            f"MAEwin={r['mae_wins']:3d}/400 "
            f"gain={r['mse_gain']:+.4f} "
            f"({r['mse_gain_pct']:+.3f}%)"
        )

    # -----------------------------------------------
    # Important operating points
    # -----------------------------------------------

    mae_safe = [
        r for r in ranking
        if (
            r["mse_mean"]
            < baseline["mse_mean"]
            and
            r["mae_mean"]
            <= baseline["mae_mean"]
        )
    ]

    stable = [
        r for r in ranking
        if r["mse_wins"] >= 300
    ]

    highly_stable = [
        r for r in ranking
        if r["mse_wins"] >= 350
    ]

    def show(title, candidates):
        print()
        print("=" * 100)
        print(title)
        print("=" * 100)

        if not candidates:
            print("NONE")
            return None

        x = min(
            candidates,
            key=lambda z: (
                z["mse_mean"],
                z["mae_mean"],
            )
        )

        print(json.dumps(x, indent=2))
        return x

    best_mae_safe = show(
        "MAE-SAFE RESULT",
        mae_safe,
    )

    best_stable = show(
        "STABLE >=300/400",
        stable,
    )

    best_highly_stable = show(
        "HIGHLY STABLE >=350/400",
        highly_stable,
    )

    print()
    print("=" * 100)
    print("BEST MSE RESULT")
    print("=" * 100)
    print(json.dumps(best, indent=2))

    np.save(
        out / "best_srr_pred_norm.npy",
        best_pred,
    )

    with open(
        out / "summary.json",
        "w",
    ) as f:
        json.dump(
            {
                "protocol": {
                    "dataset": "formal VAL400",
                    "test_used": False,
                    "raw_model":
                        "C4 step3500, sigma=11.71851402",
                    "method":
                        "Gaussian-smoothed soft-threshold "
                        "residual refinement",
                },
                "baseline": baseline,
                "best_mse": best,
                "best_mae_safe": best_mae_safe,
                "best_stable": best_stable,
                "best_highly_stable":
                    best_highly_stable,
                "ranking": ranking,
            },
            f,
            indent=2,
        )

    with open(
        out / "summary.csv",
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "smooth_sigma_px",
                "tau_mps",
                "alpha",
                "mse_mean",
                "mae_mean",
                "psnr_mean",
                "mse_wins",
                "mae_wins",
                "mse_gain",
                "mse_gain_pct",
            ],
        )

        writer.writeheader()
        writer.writerows(ranking)

    print()
    print(
        "[PASS] SRR VAL400 "
        "screening complete."
    )


if __name__ == "__main__":
    main()
