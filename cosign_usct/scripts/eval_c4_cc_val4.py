import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch as th
import matplotlib.pyplot as plt

from cc.script_util import create_model_and_diffusion


CENTER = 1502.5
SCALE = 102.5
SPEED_MIN = 1400.0
SPEED_MAX = 1605.0
DATA_RANGE = SPEED_MAX - SPEED_MIN


def norm_to_speed(x):
    return x * SCALE + CENTER


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def psnr(a, b):
    m = mse(a, b)
    if m <= 0:
        return float("inf")
    return float(
        10.0 * math.log10(
            (DATA_RANGE ** 2) / m
        )
    )


def ssim_metric(a, b):
    try:
        from skimage.metrics import structural_similarity
    except Exception as e:
        raise RuntimeError(
            "scikit-image is required for SSIM: "
            + repr(e)
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


def create_models(backbone_path, control_path, device):
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

    print("loading backbone:", backbone_path)

    backbone_state = th.load(
        backbone_path,
        map_location="cpu",
    )

    controlled_unet.load_state_dict(
        backbone_state,
        strict=True,
    )

    print("loading control:", control_path)

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

    return control_net, controlled_unet, diffusion


@th.no_grad()
def reconstruct_one_step(
    control_net,
    controlled_unet,
    diffusion,
    hint_norm,
    noise,
    device,
):
    hint = th.from_numpy(
        hint_norm[None, None].astype(np.float32)
    ).to(device)

    x_t = th.from_numpy(
        noise[None, None].astype(np.float32)
    ).to(device)

    sigma = th.full(
        (1,),
        80.0,
        device=device,
        dtype=th.float32,
    )

    _, denoised = diffusion.recon(
        controlled_unet,
        control_net,
        x_t,
        hint,
        sigma,
    )

    denoised = (
        denoised
        .clamp(-1, 1)
        .float()
        .cpu()
        .numpy()[0, 0]
    )

    return denoised


def evaluate_checkpoint(
    tag,
    backbone_path,
    control_path,
    gt_norm,
    hint_norm,
    noises,
    out_dir,
    device,
):
    print()
    print("=" * 80)
    print("EVALUATING:", tag)
    print("=" * 80)

    control_net, controlled_unet, diffusion = (
        create_models(
            backbone_path,
            control_path,
            device,
        )
    )

    preds_norm = []

    for i in range(len(gt_norm)):
        pred = reconstruct_one_step(
            control_net,
            controlled_unet,
            diffusion,
            hint_norm[i],
            noises[i],
            device,
        )

        preds_norm.append(pred)

        print(
            f"{tag} sample {i}: "
            f"range=({pred.min():.6f},"
            f"{pred.max():.6f})"
        )

    preds_norm = np.stack(
        preds_norm,
        axis=0,
    ).astype(np.float32)

    np.save(
        out_dir / f"{tag}_pred_norm.npy",
        preds_norm,
    )

    del control_net
    del controlled_unet

    th.cuda.empty_cache()

    return preds_norm


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--backbone",
        required=True,
    )

    ap.add_argument(
        "--control25",
        required=True,
    )

    ap.add_argument(
        "--control50",
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

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_norm = np.load(args.gt).astype(
        np.float32
    )

    hint_norm = np.load(args.hint).astype(
        np.float32
    )

    if gt_norm.shape != (4, 256, 256):
        raise RuntimeError(
            f"Unexpected GT shape: {gt_norm.shape}"
        )

    if hint_norm.shape != gt_norm.shape:
        raise RuntimeError(
            f"GT/hint mismatch: "
            f"{gt_norm.shape} vs "
            f"{hint_norm.shape}"
        )

    print("GT   shape =", gt_norm.shape)
    print("hint shape =", hint_norm.shape)

    # -------------------------------------------------
    # Fixed noise: exactly identical for all checkpoints.
    # -------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    noises = (
        rng.standard_normal(
            gt_norm.shape
        ).astype(np.float32)
        * 80.0
    )

    np.save(
        out_dir / "fixed_xT_sigma80.npy",
        noises,
    )

    device = th.device(
        "cuda"
        if th.cuda.is_available()
        else "cpu"
    )

    print("device =", device)

    pred25_norm = evaluate_checkpoint(
        "cc25",
        args.backbone,
        args.control25,
        gt_norm,
        hint_norm,
        noises,
        out_dir,
        device,
    )

    pred50_norm = evaluate_checkpoint(
        "cc50",
        args.backbone,
        args.control50,
        gt_norm,
        hint_norm,
        noises,
        out_dir,
        device,
    )

    # -------------------------------------------------
    # Physical-unit metrics.
    # -------------------------------------------------

    gt_speed = norm_to_speed(
        gt_norm
    )

    hint_speed = norm_to_speed(
        hint_norm
    )

    pred25_speed = norm_to_speed(
        pred25_norm
    )

    pred50_speed = norm_to_speed(
        pred50_norm
    )

    np.save(
        out_dir / "cc25_pred_speed.npy",
        pred25_speed.astype(np.float32),
    )

    np.save(
        out_dir / "cc50_pred_speed.npy",
        pred50_speed.astype(np.float32),
    )

    methods = {
        "hint": hint_speed,
        "cc25": pred25_speed,
        "cc50": pred50_speed,
    }

    rows = []

    print()
    print("=" * 80)
    print("PER-SAMPLE METRICS")
    print("=" * 80)

    for method, pred in methods.items():
        for i in range(len(gt_speed)):
            m = metric_dict(
                pred[i],
                gt_speed[i],
            )

            row = {
                "method": method,
                "sample": i,
                **m,
            }

            rows.append(row)

            print(
                f"{method:6s} "
                f"sample={i} "
                f"MSE={m['mse']:.6f} "
                f"MAE={m['mae']:.6f} "
                f"PSNR={m['psnr']:.4f} "
                f"SSIM={m['ssim']:.6f}"
            )

    with open(
        out_dir / "per_sample_metrics.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "sample",
                "mse",
                "mae",
                "psnr",
                "ssim",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {}

    print()
    print("=" * 80)
    print("MEAN ± STD")
    print("=" * 80)

    for method in methods:
        sub = [
            r
            for r in rows
            if r["method"] == method
        ]

        summary[method] = {}

        for key in [
            "mse",
            "mae",
            "psnr",
            "ssim",
        ]:
            vals = np.array(
                [r[key] for r in sub],
                dtype=np.float64,
            )

            summary[method][
                key + "_mean"
            ] = float(vals.mean())

            summary[method][
                key + "_std"
            ] = float(vals.std())

        print(
            f"{method:6s} | "
            f"MSE "
            f"{summary[method]['mse_mean']:.4f}"
            f" ± "
            f"{summary[method]['mse_std']:.4f} | "
            f"MAE "
            f"{summary[method]['mae_mean']:.4f}"
            f" ± "
            f"{summary[method]['mae_std']:.4f} | "
            f"PSNR "
            f"{summary[method]['psnr_mean']:.4f}"
            f" ± "
            f"{summary[method]['psnr_std']:.4f} | "
            f"SSIM "
            f"{summary[method]['ssim_mean']:.4f}"
            f" ± "
            f"{summary[method]['ssim_std']:.4f}"
        )

    # wins relative to hint
    for tag in ["cc25", "cc50"]:
        mse_wins = 0
        mae_wins = 0
        ssim_wins = 0

        for i in range(len(gt_speed)):
            h = metric_dict(
                hint_speed[i],
                gt_speed[i],
            )

            c = metric_dict(
                methods[tag][i],
                gt_speed[i],
            )

            mse_wins += int(
                c["mse"] < h["mse"]
            )
            mae_wins += int(
                c["mae"] < h["mae"]
            )
            ssim_wins += int(
                c["ssim"] > h["ssim"]
            )

        summary[tag]["mse_wins_vs_hint"] = (
            mse_wins
        )
        summary[tag]["mae_wins_vs_hint"] = (
            mae_wins
        )
        summary[tag]["ssim_wins_vs_hint"] = (
            ssim_wins
        )

        print(
            f"{tag} wins vs hint: "
            f"MSE {mse_wins}/4, "
            f"MAE {mae_wins}/4, "
            f"SSIM {ssim_wins}/4"
        )

    with open(
        out_dir / "summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # -------------------------------------------------
    # Visualization.
    # -------------------------------------------------

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(13, 13),
    )

    column_data = [
        ("GT", gt_speed),
        ("InversionNet hint", hint_speed),
        ("C4 CC-25", pred25_speed),
        ("C4 CC-50", pred50_speed),
    ]

    for r in range(4):
        for c, (name, arr) in enumerate(
            column_data
        ):
            ax = axes[r, c]

            im = ax.imshow(
                arr[r],
                cmap="inferno",
                vmin=SPEED_MIN,
                vmax=SPEED_MAX,
            )

            ax.set_title(
                f"sample {r} | {name}",
                fontsize=9,
            )

            ax.axis("off")

    plt.tight_layout()

    fig.savefig(
        out_dir /
        "compare_gt_hint_cc25_cc50.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("[PASS] C4 VAL4 evaluation complete")
    print("output =", out_dir)


if __name__ == "__main__":
    main()
