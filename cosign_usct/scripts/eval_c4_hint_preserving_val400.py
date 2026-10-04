import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch as th

from cc.script_util import create_model_and_diffusion
from skimage.metrics import structural_similarity


CENTER = 1502.5
SCALE = 102.5

SPEED_MIN = 1400.0
SPEED_MAX = 1605.0
DATA_RANGE = SPEED_MAX - SPEED_MIN


def norm_to_speed(x):
    return x * SCALE + CENTER


def metrics(pred_norm, gt_norm):
    pred_norm = np.clip(
        pred_norm,
        -1.0,
        1.0,
    )

    gt_norm = np.clip(
        gt_norm,
        -1.0,
        1.0,
    )

    pred = norm_to_speed(
        pred_norm
    )

    gt = norm_to_speed(
        gt_norm
    )

    diff = pred - gt

    mse = float(
        np.mean(diff ** 2)
    )

    mae = float(
        np.mean(np.abs(diff))
    )

    psnr = float(
        10.0
        * math.log10(
            DATA_RANGE ** 2 / mse
        )
    )

    ssim = float(
        structural_similarity(
            gt,
            pred,
            data_range=DATA_RANGE,
        )
    )

    return {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim,
    }


def create_models(
    backbone,
    control,
    device,
):
    control_net, model, diffusion = (
        create_model_and_diffusion(
            image_size=256,
            class_cond=False,
            learn_sigma=False,
            num_channels=256,
            num_res_blocks=2,
            channel_mult="",
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            attention_resolutions="32,16,8",
            dropout=0.0,
            use_checkpoint=False,
            use_scale_shift_norm=False,
            resblock_updown=True,
            use_fp16=True,
            use_new_attention_order=False,
            weight_schedule="uniform",
            sigma_min=0.002,
            sigma_max=80.0,
            loss_norm="l2",
            loss_type="recon",
            distillation=True,
            control=True,
            in_channels=1,
        )
    )

    model.load_state_dict(
        th.load(
            backbone,
            map_location="cpu",
        ),
        strict=True,
    )

    control_net.load_state_dict(
        th.load(
            control,
            map_location="cpu",
        ),
        strict=True,
    )

    model.to(device)
    control_net.to(device)

    model.convert_to_fp16()
    control_net.convert_to_fp16()

    model.eval()
    control_net.eval()

    return (
        control_net,
        model,
        diffusion,
    )


@th.no_grad()
def refine_batch(
    hint_np,
    noise_np,
    sigma_value,
    control_net,
    model,
    diffusion,
    device,
):
    hint = th.from_numpy(
        hint_np[:, None]
    ).float().to(device)

    noise = th.from_numpy(
        noise_np[:, None]
    ).float().to(device)

    sigma = th.full(
        (len(hint_np),),
        sigma_value,
        dtype=th.float32,
        device=device,
    )

    # --------------------------------------------------
    # KEY:
    # preserve InversionNet initialization.
    #
    # x_t = hint + sigma * epsilon
    # --------------------------------------------------

    x_t = (
        hint
        + sigma[:, None, None, None]
        * noise
    )

    _, pred = diffusion.recon(
        model,
        control_net,
        x_t,
        hint,
        sigma,
    )

    return (
        pred.float()
        .cpu()
        .numpy()[:, 0]
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--backbone",
        required=True,
    )

    ap.add_argument(
        "--control",
        required=True,
    )

    ap.add_argument(
        "--gt",
        required=True,
    )

    ap.add_argument(
        "--hint",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260927,
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    args = ap.parse_args()

    out = Path(args.out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt = np.load(
        args.gt
    ).astype(np.float32)

    hint = np.load(
        args.hint
    ).astype(np.float32)

    if gt.shape != hint.shape:
        raise RuntimeError(
            f"shape mismatch: "
            f"{gt.shape} vs "
            f"{hint.shape}"
        )

    if gt.shape != (
        400,
        256,
        256,
    ):
        raise RuntimeError(
            f"expected VAL400, got "
            f"{gt.shape}"
        )

    n = len(gt)

    device = th.device("cuda")

    (
        control_net,
        model,
        diffusion,
    ) = create_models(
        args.backbone,
        args.control,
        device,
    )

    # Exact representative Karras-grid values
    # already used in C3 validation.
    sigmas = [
        11.71851402,
        3.51969883,
        0.82294137,
        0.13119736,
        0.01081205,
        0.00200000,
    ]

    # alpha=0 is EXACT InversionNet baseline.
    alphas = [
        0.0,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
        0.75,
        1.00,
    ]

    rng = np.random.default_rng(
        args.seed
    )

    # Same epsilon for all sigma values:
    # paired comparison.
    noise = rng.standard_normal(
        gt.shape
    ).astype(np.float32)

    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------

    baseline_vals = [
        metrics(
            hint[i],
            gt[i],
        )
        for i in range(n)
    ]

    baseline = {}

    for key in [
        "mse",
        "mae",
        "psnr",
        "ssim",
    ]:
        a = np.asarray(
            [
                x[key]
                for x
                in baseline_vals
            ],
            dtype=np.float64,
        )

        baseline[
            key + "_mean"
        ] = float(a.mean())

        baseline[
            key + "_std"
        ] = float(a.std())

    print()
    print("=" * 100)
    print("INVERSIONNET VAL400 BASELINE")
    print("=" * 100)

    print(
        json.dumps(
            baseline,
            indent=2,
        )
    )

    rows = []
    per_sample = []

    # --------------------------------------------------
    # sigma sweep
    # --------------------------------------------------

    for sigma_value in sigmas:

        print()
        print("=" * 100)
        print(
            "sigma =",
            sigma_value,
        )
        print("=" * 100)

        preds = []

        for start in range(
            0,
            n,
            args.batch_size,
        ):
            end = min(
                n,
                start
                + args.batch_size,
            )

            p = refine_batch(
                hint[start:end],
                noise[start:end],
                sigma_value,
                control_net,
                model,
                diffusion,
                device,
            )

            preds.append(p)

        pred = np.concatenate(
            preds,
            axis=0,
        ).astype(np.float32)

        np.save(
            out /
            (
                f"raw_pred_sigma_"
                f"{sigma_value:.8f}.npy"
            ),
            pred,
        )

        # ----------------------------------------------
        # residual interpolation sweep
        # x_ref = hint + alpha * (pred - hint)
        # ----------------------------------------------

        for alpha in alphas:

            refined = (
                hint
                + alpha
                * (pred - hint)
            )

            refined = np.clip(
                refined,
                -1.0,
                1.0,
            )

            vals = []

            mse_wins = 0
            mae_wins = 0
            ssim_wins = 0

            for i in range(n):

                m = metrics(
                    refined[i],
                    gt[i],
                )

                h = baseline_vals[i]

                mse_win = (
                    m["mse"]
                    <
                    h["mse"]
                )

                mae_win = (
                    m["mae"]
                    <
                    h["mae"]
                )

                ssim_win = (
                    m["ssim"]
                    >
                    h["ssim"]
                )

                mse_wins += int(
                    mse_win
                )

                mae_wins += int(
                    mae_win
                )

                ssim_wins += int(
                    ssim_win
                )

                vals.append(m)

                per_sample.append({
                    "sigma":
                        sigma_value,

                    "alpha":
                        alpha,

                    "sample":
                        i,

                    "mse":
                        m["mse"],

                    "mae":
                        m["mae"],

                    "psnr":
                        m["psnr"],

                    "ssim":
                        m["ssim"],

                    "hint_mse":
                        h["mse"],

                    "mse_win":
                        int(mse_win),

                    "mae_win":
                        int(mae_win),

                    "ssim_win":
                        int(ssim_win),
                })

            summary = {
                "sigma":
                    sigma_value,

                "alpha":
                    alpha,

                "mse_mean":
                    float(
                        np.mean(
                            [
                                v["mse"]
                                for v
                                in vals
                            ]
                        )
                    ),

                "mae_mean":
                    float(
                        np.mean(
                            [
                                v["mae"]
                                for v
                                in vals
                            ]
                        )
                    ),

                "psnr_mean":
                    float(
                        np.mean(
                            [
                                v["psnr"]
                                for v
                                in vals
                            ]
                        )
                    ),

                "ssim_mean":
                    float(
                        np.mean(
                            [
                                v["ssim"]
                                for v
                                in vals
                            ]
                        )
                    ),

                "mse_wins":
                    mse_wins,

                "mae_wins":
                    mae_wins,

                "ssim_wins":
                    ssim_wins,
            }

            summary[
                "mse_gain_vs_hint"
            ] = (
                baseline[
                    "mse_mean"
                ]
                -
                summary[
                    "mse_mean"
                ]
            )

            rows.append(
                summary
            )

            print(
                f"sigma="
                f"{sigma_value:11.8f} "
                f"alpha="
                f"{alpha:4.2f} | "
                f"MSE="
                f"{summary['mse_mean']:9.4f} "
                f"MAE="
                f"{summary['mae_mean']:7.4f} "
                f"SSIM="
                f"{summary['ssim_mean']:.5f} "
                f"| MSEwin="
                f"{mse_wins:3d}/{n} "
                f"| gain="
                f"{summary['mse_gain_vs_hint']:+.4f}"
            )

    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    ranked = sorted(
        rows,
        key=lambda x: (
            x["mse_mean"],
            x["mae_mean"],
        )
    )

    print()
    print("=" * 120)
    print(
        "HINT-PRESERVING VAL400 RANKING"
    )
    print("=" * 120)

    for rank, x in enumerate(
        ranked[:20],
        start=1,
    ):
        print(
            f"#{rank:02d} "
            f"sigma="
            f"{x['sigma']:11.8f} "
            f"alpha="
            f"{x['alpha']:4.2f} "
            f"MSE="
            f"{x['mse_mean']:9.4f} "
            f"MAE="
            f"{x['mae_mean']:7.4f} "
            f"PSNR="
            f"{x['psnr_mean']:7.3f} "
            f"SSIM="
            f"{x['ssim_mean']:.5f} "
            f"MSEwin="
            f"{x['mse_wins']:3d}/{n} "
            f"gain="
            f"{x['mse_gain_vs_hint']:+.4f}"
        )

    best = ranked[0]

    print()
    print(
        "BEST =",
        best,
    )

    print()
    print(
        "BASELINE MSE =",
        baseline["mse_mean"],
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    with open(
        out / "summary.json",
        "w",
    ) as f:
        json.dump(
            {
                "protocol": {
                    "dataset":
                        "formal VAL400",

                    "x_t":
                        "hint + sigma * epsilon",

                    "refinement":
                        "hint + alpha * "
                        "(C4_output - hint)",

                    "seed":
                        args.seed,

                    "test_used":
                        False,
                },

                "baseline":
                    baseline,

                "ranking":
                    ranked,

                "best":
                    best,
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
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    with open(
        out /
        "per_sample.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                per_sample[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            per_sample
        )

    print()
    print(
        "[PASS] hint-preserving "
        "VAL400 sweep complete."
    )


if __name__ == "__main__":
    main()
