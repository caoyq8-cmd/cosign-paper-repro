import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch as th

from cc.script_util import create_model_and_diffusion


CENTER = 1502.5
SCALE = 102.5

SPEED_MIN = 1400.0
SPEED_MAX = 1605.0
DATA_RANGE = SPEED_MAX - SPEED_MIN


def norm_to_speed(x):
    return x * SCALE + CENTER


def mse(a, b):
    return float(
        np.mean((a - b) ** 2)
    )


def mae(a, b):
    return float(
        np.mean(np.abs(a - b))
    )


def psnr(a, b):
    m = mse(a, b)

    if m <= 0:
        return float("inf")

    return float(
        10.0 * math.log10(
            DATA_RANGE ** 2 / m
        )
    )


def ssim_metric(a, b):
    from skimage.metrics import (
        structural_similarity,
    )

    return float(
        structural_similarity(
            a,
            b,
            data_range=DATA_RANGE,
        )
    )


def metric_dict(pred, gt):
    return {
        "mse": mse(pred, gt),
        "mae": mae(pred, gt),
        "psnr": psnr(pred, gt),
        "ssim": ssim_metric(pred, gt),
    }


def create_models(
    backbone_path,
    control_path,
    device,
):
    control_net, controlled_unet, diffusion = (
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

    backbone_state = th.load(
        backbone_path,
        map_location="cpu",
    )

    controlled_unet.load_state_dict(
        backbone_state,
        strict=True,
    )

    control_state = th.load(
        control_path,
        map_location="cpu",
    )

    control_net.load_state_dict(
        control_state,
        strict=True,
    )

    controlled_unet.to(device)
    control_net.to(device)

    controlled_unet.convert_to_fp16()
    control_net.convert_to_fp16()

    controlled_unet.eval()
    control_net.eval()

    return (
        control_net,
        controlled_unet,
        diffusion,
    )


@th.no_grad()
def reconstruct_one(
    xT,
    hint,
    control_net,
    controlled_unet,
    diffusion,
    device,
):
    xt = th.from_numpy(
        xT[None, None].astype(
            np.float32
        )
    ).to(device)

    h = th.from_numpy(
        hint[None, None].astype(
            np.float32
        )
    ).to(device)

    sigma = th.full(
        (1,),
        80.0,
        dtype=th.float32,
        device=device,
    )

    _, pred = diffusion.recon(
        controlled_unet,
        control_net,
        xt,
        h,
        sigma,
    )

    return (
        pred
        .clamp(-1, 1)
        .float()
        .cpu()
        .numpy()[0, 0]
    )


def evaluate_checkpoint(
    step,
    backbone,
    control,
    gt_norm,
    hint_norm,
    fixed_xT,
    device,
    out,
):
    print()
    print("=" * 90)
    print(
        f"REAL ONE-STEP VAL20 | "
        f"checkpoint={step}"
    )
    print("=" * 90)

    (
        control_net,
        controlled_unet,
        diffusion,
    ) = create_models(
        backbone,
        control,
        device,
    )

    preds = []

    for i in range(len(gt_norm)):

        p = reconstruct_one(
            fixed_xT[i],
            hint_norm[i],
            control_net,
            controlled_unet,
            diffusion,
            device,
        )

        preds.append(p)

        print(
            f"step={step:4d} "
            f"sample={i:02d} "
            f"range=("
            f"{p.min():.5f},"
            f"{p.max():.5f})"
        )

    preds = np.stack(
        preds
    ).astype(np.float32)

    np.save(
        out /
        f"step{step}_pred_norm.npy",
        preds,
    )

    del control_net
    del controlled_unet
    del diffusion

    if th.cuda.is_available():
        th.cuda.empty_cache()

    return preds


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--backbone",
        required=True,
    )

    ap.add_argument(
        "--run",
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
        default=20260904,
    )

    ap.add_argument(
        "--steps",
        default=(
            "500,1000,1500,2000,2500,"
            "3000,3500,4000,4500,5000"
        ),
    )

    args = ap.parse_args()

    out = Path(args.out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    run = Path(args.run)

    steps = [
        int(x)
        for x in args.steps.split(",")
        if x.strip()
    ]

    gt_norm = np.load(
        args.gt
    ).astype(np.float32)

    hint_norm = np.load(
        args.hint
    ).astype(np.float32)

    if gt_norm.shape != hint_norm.shape:
        raise RuntimeError(
            f"GT/hint mismatch: "
            f"{gt_norm.shape} vs "
            f"{hint_norm.shape}"
        )

    if gt_norm.ndim != 3:
        raise RuntimeError(
            f"Expected [N,H,W], "
            f"got {gt_norm.shape}"
        )

    print(
        "VAL shape =",
        gt_norm.shape,
    )

    # --------------------------------------------------
    # Fixed PURE-NOISE x_T.
    #
    # Important:
    # NO GT enters x_T construction.
    # --------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    fixed_xT = (
        rng.standard_normal(
            gt_norm.shape
        ).astype(np.float32)
        * 80.0
    )

    np.save(
        out / "fixed_pure_noise_xT_sigma80.npy",
        fixed_xT,
    )

    print(
        "xT construction = "
        "pure Gaussian noise * sigma_max"
    )

    device = th.device(
        "cuda"
        if th.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device,
    )

    gt_speed = norm_to_speed(
        gt_norm
    )

    hint_speed = norm_to_speed(
        hint_norm
    )

    # --------------------------------------------------
    # Baseline metrics: InversionNet hint
    # --------------------------------------------------

    hint_metrics = []

    for i in range(len(gt_speed)):
        hint_metrics.append(
            metric_dict(
                hint_speed[i],
                gt_speed[i],
            )
        )

    hint_summary = {}

    for key in [
        "mse",
        "mae",
        "psnr",
        "ssim",
    ]:
        v = np.array(
            [
                x[key]
                for x in hint_metrics
            ],
            dtype=np.float64,
        )

        hint_summary[
            key + "_mean"
        ] = float(v.mean())

        hint_summary[
            key + "_std"
        ] = float(v.std())

    print()
    print("=" * 90)
    print("INVERSIONNET HINT BASELINE")
    print("=" * 90)

    print(
        f"MSE  = "
        f"{hint_summary['mse_mean']:.6f}"
        f" ± "
        f"{hint_summary['mse_std']:.6f}"
    )

    print(
        f"MAE  = "
        f"{hint_summary['mae_mean']:.6f}"
        f" ± "
        f"{hint_summary['mae_std']:.6f}"
    )

    print(
        f"PSNR = "
        f"{hint_summary['psnr_mean']:.4f}"
        f" ± "
        f"{hint_summary['psnr_std']:.4f}"
    )

    print(
        f"SSIM = "
        f"{hint_summary['ssim_mean']:.6f}"
        f" ± "
        f"{hint_summary['ssim_std']:.6f}"
    )

    summaries = []
    per_sample_rows = []

    # --------------------------------------------------
    # Evaluate all C4 checkpoints
    # --------------------------------------------------

    for step in steps:

        control = (
            run /
            f"model{step:06d}.pt"
        )

        if not control.exists():
            print(
                "[WARN] missing checkpoint:",
                control,
            )
            continue

        pred_norm = evaluate_checkpoint(
            step,
            args.backbone,
            str(control),
            gt_norm,
            hint_norm,
            fixed_xT,
            device,
            out,
        )

        pred_speed = norm_to_speed(
            pred_norm
        )

        vals = []

        mse_wins = 0
        mae_wins = 0
        ssim_wins = 0

        for i in range(len(gt_speed)):

            m = metric_dict(
                pred_speed[i],
                gt_speed[i],
            )

            vals.append(m)

            hm = hint_metrics[i]

            mse_win = (
                m["mse"]
                <
                hm["mse"]
            )

            mae_win = (
                m["mae"]
                <
                hm["mae"]
            )

            ssim_win = (
                m["ssim"]
                >
                hm["ssim"]
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

            per_sample_rows.append({
                "step": step,
                "sample": i,
                "mse": m["mse"],
                "mae": m["mae"],
                "psnr": m["psnr"],
                "ssim": m["ssim"],
                "hint_mse":
                    hm["mse"],
                "hint_mae":
                    hm["mae"],
                "hint_ssim":
                    hm["ssim"],
                "mse_win_vs_hint":
                    int(mse_win),
                "mae_win_vs_hint":
                    int(mae_win),
                "ssim_win_vs_hint":
                    int(ssim_win),
            })

        summary = {
            "step": step,
            "n": len(vals),
        }

        for key in [
            "mse",
            "mae",
            "psnr",
            "ssim",
        ]:

            a = np.array(
                [
                    v[key]
                    for v in vals
                ],
                dtype=np.float64,
            )

            summary[
                key + "_mean"
            ] = float(
                a.mean()
            )

            summary[
                key + "_std"
            ] = float(
                a.std()
            )

        summary[
            "mse_wins_vs_hint"
        ] = mse_wins

        summary[
            "mae_wins_vs_hint"
        ] = mae_wins

        summary[
            "ssim_wins_vs_hint"
        ] = ssim_wins

        summary[
            "mse_gain_vs_hint"
        ] = (
            hint_summary[
                "mse_mean"
            ]
            -
            summary[
                "mse_mean"
            ]
        )

        summary[
            "mae_gain_vs_hint"
        ] = (
            hint_summary[
                "mae_mean"
            ]
            -
            summary[
                "mae_mean"
            ]
        )

        summaries.append(
            summary
        )

        print()
        print(
            f"STEP {step}"
        )

        print(
            f"MSE  "
            f"{summary['mse_mean']:.6f}"
            f" ± "
            f"{summary['mse_std']:.6f}"
            f" | wins="
            f"{mse_wins}/{len(vals)}"
        )

        print(
            f"MAE  "
            f"{summary['mae_mean']:.6f}"
            f" ± "
            f"{summary['mae_std']:.6f}"
            f" | wins="
            f"{mae_wins}/{len(vals)}"
        )

        print(
            f"PSNR "
            f"{summary['psnr_mean']:.4f}"
            f" ± "
            f"{summary['psnr_std']:.4f}"
        )

        print(
            f"SSIM "
            f"{summary['ssim_mean']:.6f}"
            f" ± "
            f"{summary['ssim_std']:.6f}"
            f" | wins="
            f"{ssim_wins}/{len(vals)}"
        )

    # --------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------

    summary_csv = (
        out /
        "real_onestep_val20_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                summaries[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            summaries
        )

    sample_csv = (
        out /
        "real_onestep_val20_per_sample.csv"
    )

    with sample_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                per_sample_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            per_sample_rows
        )

    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    print()
    print("=" * 120)
    print("REAL ONE-STEP VAL20 CHECKPOINT RANKING")
    print("=" * 120)

    print(
        f"{'step':>6} "
        f"{'MSE':>12} "
        f"{'MAE':>10} "
        f"{'PSNR':>9} "
        f"{'SSIM':>9} "
        f"{'MSEwin':>8} "
        f"{'MAEwin':>8} "
        f"{'SSIMwin':>9} "
        f"{'MSEgain':>12}"
    )

    for s in sorted(
        summaries,
        key=lambda z:
            z["mse_mean"],
    ):

        print(
            f"{s['step']:6d} "
            f"{s['mse_mean']:12.5f} "
            f"{s['mae_mean']:10.5f} "
            f"{s['psnr_mean']:9.4f} "
            f"{s['ssim_mean']:9.5f} "
            f"{s['mse_wins_vs_hint']:5d}/{s['n']:<2d} "
            f"{s['mae_wins_vs_hint']:5d}/{s['n']:<2d} "
            f"{s['ssim_wins_vs_hint']:5d}/{s['n']:<2d} "
            f"{s['mse_gain_vs_hint']:+12.5f}"
        )

    best_mse = min(
        summaries,
        key=lambda z:
            z["mse_mean"],
    )

    best_mae = min(
        summaries,
        key=lambda z:
            z["mae_mean"],
    )

    best_ssim = max(
        summaries,
        key=lambda z:
            z["ssim_mean"],
    )

    print()
    print(
        "best MSE checkpoint  =",
        best_mse["step"],
    )

    print(
        "best MAE checkpoint  =",
        best_mae["step"],
    )

    print(
        "best SSIM checkpoint =",
        best_ssim["step"],
    )

    with open(
        out /
        "real_onestep_val20_summary.json",
        "w",
    ) as f:

        json.dump(
            {
                "hint":
                    hint_summary,

                "checkpoints":
                    summaries,

                "best_mse_step":
                    best_mse[
                        "step"
                    ],

                "best_mae_step":
                    best_mae[
                        "step"
                    ],

                "best_ssim_step":
                    best_ssim[
                        "step"
                    ],
            },
            f,
            indent=2,
        )

    print()
    print(
        "[PASS] REAL one-step "
        "VAL20 evaluation complete"
    )

    print(
        "summary CSV =",
        summary_csv,
    )


if __name__ == "__main__":
    main()
