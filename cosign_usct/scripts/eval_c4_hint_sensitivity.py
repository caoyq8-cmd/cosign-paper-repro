import argparse
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


def metrics(pred, gt):
    from skimage.metrics import structural_similarity

    mse = float(np.mean((pred - gt) ** 2))
    mae = float(np.mean(np.abs(pred - gt)))
    psnr = float(
        10 * math.log10(DATA_RANGE**2 / mse)
    )
    ssim = float(
        structural_similarity(
            pred,
            gt,
            data_range=DATA_RANGE,
        )
    )

    return {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim,
    }


def create_models(backbone, control, device):
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

    controlled_unet.load_state_dict(
        th.load(backbone, map_location="cpu"),
        strict=True,
    )

    control_net.load_state_dict(
        th.load(control, map_location="cpu"),
        strict=True,
    )

    control_net.to(device)
    controlled_unet.to(device)

    control_net.convert_to_fp16()
    controlled_unet.convert_to_fp16()

    control_net.eval()
    controlled_unet.eval()

    return control_net, controlled_unet, diffusion


@th.no_grad()
def reconstruct(
    x_t,
    hint,
    control_net,
    controlled_unet,
    diffusion,
    device,
):
    xt = th.from_numpy(
        x_t[None, None].astype(np.float32)
    ).to(device)

    h = th.from_numpy(
        hint[None, None].astype(np.float32)
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
        pred.clamp(-1, 1)
        .float()
        .cpu()
        .numpy()[0, 0]
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--backbone", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--hint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--seed",
        type=int,
        default=20260904,
    )

    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt = np.load(args.gt).astype(np.float32)
    hint = np.load(args.hint).astype(np.float32)

    if gt.shape != (4, 256, 256):
        raise RuntimeError(gt.shape)

    if hint.shape != gt.shape:
        raise RuntimeError(
            f"GT/hint mismatch: {gt.shape} {hint.shape}"
        )

    zero_hint = np.zeros_like(hint)

    # rotate sample identities
    shuffled_hint = np.roll(
        hint,
        shift=1,
        axis=0,
    )

    rng = np.random.default_rng(args.seed)

    # exactly the same sigma=80 x_T for all hint variants
    xT = (
        rng.standard_normal(gt.shape)
        .astype(np.float32)
        * 80.0
    )

    device = th.device("cuda")

    control_net, controlled_unet, diffusion = (
        create_models(
            args.backbone,
            args.control,
            device,
        )
    )

    variants = {
        "correct": hint,
        "zero": zero_hint,
        "shuffled": shuffled_hint,
    }

    pred_norm = {}

    for name, hints in variants.items():
        preds = []

        print("\n====", name, "====")

        for i in range(len(gt)):
            p = reconstruct(
                xT[i],
                hints[i],
                control_net,
                controlled_unet,
                diffusion,
                device,
            )

            preds.append(p)

            print(
                f"sample {i}: "
                f"range=({p.min():.6f},"
                f"{p.max():.6f})"
            )

        pred_norm[name] = np.stack(
            preds
        ).astype(np.float32)

        np.save(
            out / f"pred_{name}_norm.npy",
            pred_norm[name],
        )

    gt_speed = norm_to_speed(gt)

    summary = {}

    print("\n" + "=" * 90)
    print("METRICS VS GT")
    print("=" * 90)

    for name in variants:
        pred_speed = norm_to_speed(
            pred_norm[name]
        )

        vals = [
            metrics(
                pred_speed[i],
                gt_speed[i],
            )
            for i in range(len(gt))
        ]

        summary[name] = {}

        for key in [
            "mse",
            "mae",
            "psnr",
            "ssim",
        ]:
            a = np.array(
                [v[key] for v in vals]
            )

            summary[name][
                key + "_mean"
            ] = float(a.mean())

            summary[name][
                key + "_std"
            ] = float(a.std())

        print(
            f"{name:9s} | "
            f"MSE={summary[name]['mse_mean']:.4f} | "
            f"MAE={summary[name]['mae_mean']:.4f} | "
            f"PSNR={summary[name]['psnr_mean']:.4f} | "
            f"SSIM={summary[name]['ssim_mean']:.4f}"
        )

    # output sensitivity
    corr = pred_norm["correct"]

    for other in [
        "zero",
        "shuffled",
    ]:
        d_norm = np.abs(
            corr - pred_norm[other]
        )

        d_mps = d_norm * SCALE

        summary[
            f"correct_vs_{other}"
        ] = {
            "mean_abs_diff_norm":
                float(d_norm.mean()),
            "max_abs_diff_norm":
                float(d_norm.max()),
            "mean_abs_diff_mps":
                float(d_mps.mean()),
            "max_abs_diff_mps":
                float(d_mps.max()),
        }

        print()
        print(
            f"correct vs {other}:"
        )
        print(
            " mean abs output diff [norm] =",
            float(d_norm.mean()),
        )
        print(
            " mean abs output diff [m/s]  =",
            float(d_mps.mean()),
        )
        print(
            " max abs output diff [m/s]   =",
            float(d_mps.max()),
        )

    # Per-sample correct vs shuffled wins
    correct_wins = 0

    corr_speed = norm_to_speed(
        pred_norm["correct"]
    )

    shuf_speed = norm_to_speed(
        pred_norm["shuffled"]
    )

    for i in range(len(gt)):
        mc = metrics(
            corr_speed[i],
            gt_speed[i],
        )
        ms = metrics(
            shuf_speed[i],
            gt_speed[i],
        )

        win = mc["mse"] < ms["mse"]
        correct_wins += int(win)

        print(
            f"sample {i}: "
            f"correct MSE={mc['mse']:.4f}, "
            f"shuffled MSE={ms['mse']:.4f}, "
            f"correct_win={win}"
        )

    summary[
        "correct_mse_wins_vs_shuffled"
    ] = correct_wins

    print(
        "\ncorrect MSE wins vs shuffled =",
        f"{correct_wins}/4",
    )

    with open(
        out / "summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # visualization
    fig, axes = plt.subplots(
        4,
        5,
        figsize=(16, 13),
    )

    columns = [
        ("GT", gt_speed),
        ("Correct hint", norm_to_speed(hint)),
        (
            "Pred | correct",
            norm_to_speed(pred_norm["correct"]),
        ),
        (
            "Pred | zero",
            norm_to_speed(pred_norm["zero"]),
        ),
        (
            "Pred | shuffled",
            norm_to_speed(pred_norm["shuffled"]),
        ),
    ]

    for r in range(4):
        for c, (title, arr) in enumerate(columns):
            ax = axes[r, c]

            ax.imshow(
                arr[r],
                cmap="inferno",
                vmin=SPEED_MIN,
                vmax=SPEED_MAX,
            )

            ax.set_title(
                f"sample {r} | {title}",
                fontsize=9,
            )

            ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        out / "hint_sensitivity.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("[PASS] hint sensitivity evaluation complete")


if __name__ == "__main__":
    main()
