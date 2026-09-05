import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch as th

from cc.script_util import create_model_and_diffusion
from cc.karras_diffusion import control_sample
from cc.random_util import get_generator


CENTER = 1502.5
SCALE = 102.5

SPEED_MIN = 1400.0
SPEED_MAX = 1605.0
DATA_RANGE = SPEED_MAX - SPEED_MIN


def norm_to_speed(x):
    return x * SCALE + CENTER


def metric_dict(pred, gt):
    from skimage.metrics import structural_similarity

    mse = float(np.mean((pred - gt) ** 2))
    mae = float(np.mean(np.abs(pred - gt)))

    psnr = float(
        10.0 * math.log10(
            DATA_RANGE**2 / mse
        )
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

    return control_net, model, diffusion


@th.no_grad()
def sample_one(
    hint_np,
    sample_index,
    num_samples,
    seed,
    ts,
    control_net,
    model,
    diffusion,
    device,
):

    hint = th.from_numpy(
        hint_np[None, None].astype(
            np.float32
        )
    ).to(device)

    # Official generator.
    #
    # Fresh generator for every configuration makes
    # the initial x_T identical across NFE settings.
    generator = get_generator(
        "determ-indiv",
        num_samples=num_samples,
        seed=seed,
    )

    generator.set_done_samples(
        sample_index
    )

    # multistep currently only consumes hint;
    # these keys are nevertheless required by
    # official control_sample().
    condition_args = {
        "hint": hint,
        "y_n": th.zeros_like(hint),
        "measurement_cond_fn": None,
    }

    sample, _ = control_sample(
        diffusion=diffusion,
        control_net=control_net,
        controlled_unet=model,
        shape=(1, 1, 256, 256),
        steps=40,
        clip_denoised=True,
        progress=False,
        callback=None,
        model_kwargs={},
        device=device,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        sampler="multistep",
        generator=generator,
        ts=tuple(ts),
        condition_args=condition_args,
    )

    return (
        sample.float()
        .cpu()
        .numpy()[0, 0]
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
        default=20260904,
    )

    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_norm = np.load(
        args.gt
    ).astype(np.float32)

    hint_norm = np.load(
        args.hint
    ).astype(np.float32)

    if gt_norm.shape != hint_norm.shape:
        raise RuntimeError(
            f"shape mismatch: "
            f"{gt_norm.shape} vs "
            f"{hint_norm.shape}"
        )

    n = len(gt_norm)

    print("VAL samples =", n)

    device = th.device(
        "cuda"
        if th.cuda.is_available()
        else "cpu"
    )

    (
        control_net,
        model,
        diffusion,
    ) = create_models(
        args.backbone,
        args.control,
        device,
    )

    # ----------------------------------------------------
    # Karras-index schedules.
    #
    # NFE = len(ts) - 1
    #
    # "2_official" reproduces the example schedule
    # documented by CoSIGN: 0,17,39.
    # Other schedules use approximately uniform spacing
    # over the 40-point Karras index grid.
    # ----------------------------------------------------

    schedules = {
        "1step": [
            0, 39,
        ],

        "2step_uniform": [
            0, 20, 39,
        ],

        "2step_official": [
            0, 17, 39,
        ],

        "3step": [
            0, 13, 26, 39,
        ],

        "5step": [
            0, 8, 16, 23, 31, 39,
        ],
    }

    gt_speed = norm_to_speed(
        gt_norm
    )

    hint_speed = norm_to_speed(
        hint_norm
    )

    # Hint baseline.
    hint_metrics = [
        metric_dict(
            hint_speed[i],
            gt_speed[i],
        )
        for i in range(n)
    ]

    hint_mse = np.mean(
        [x["mse"] for x in hint_metrics]
    )

    hint_mae = np.mean(
        [x["mae"] for x in hint_metrics]
    )

    hint_ssim = np.mean(
        [x["ssim"] for x in hint_metrics]
    )

    print()
    print("=" * 110)
    print("INVERSIONNET HINT")
    print("=" * 110)

    print(
        f"MSE={hint_mse:.6f} "
        f"MAE={hint_mae:.6f} "
        f"SSIM={hint_ssim:.6f}"
    )

    summaries = []
    sample_rows = []

    for tag, ts in schedules.items():

        print()
        print("=" * 110)
        print(
            f"{tag} | "
            f"NFE={len(ts)-1} | "
            f"ts={ts}"
        )
        print("=" * 110)

        preds = []

        for i in range(n):

            p = sample_one(
                hint_np=hint_norm[i],
                sample_index=i,
                num_samples=n,
                seed=args.seed,
                ts=ts,
                control_net=control_net,
                model=model,
                diffusion=diffusion,
                device=device,
            )

            preds.append(p)

            print(
                f"{tag:16s} "
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
            f"{tag}_pred_norm.npy",
            preds,
        )

        pred_speed = norm_to_speed(
            preds
        )

        vals = []

        mse_wins = 0
        mae_wins = 0
        ssim_wins = 0

        for i in range(n):

            m = metric_dict(
                pred_speed[i],
                gt_speed[i],
            )

            h = hint_metrics[i]

            mse_win = (
                m["mse"] < h["mse"]
            )

            mae_win = (
                m["mae"] < h["mae"]
            )

            ssim_win = (
                m["ssim"] > h["ssim"]
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

            sample_rows.append({
                "schedule": tag,
                "nfe": len(ts) - 1,
                "ts": ",".join(
                    map(str, ts)
                ),
                "sample": i,
                "mse": m["mse"],
                "mae": m["mae"],
                "psnr": m["psnr"],
                "ssim": m["ssim"],
                "hint_mse": h["mse"],
                "mse_win_vs_hint":
                    int(mse_win),
                "mae_win_vs_hint":
                    int(mae_win),
                "ssim_win_vs_hint":
                    int(ssim_win),
            })

        summary = {
            "schedule": tag,
            "nfe": len(ts) - 1,
            "ts": ",".join(
                map(str, ts)
            ),
        }

        for key in [
            "mse",
            "mae",
            "psnr",
            "ssim",
        ]:

            a = np.asarray(
                [
                    x[key]
                    for x in vals
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
        ] = float(
            hint_mse
            -
            summary["mse_mean"]
        )

        summaries.append(
            summary
        )

        print()
        print(
            f"{tag}: "
            f"MSE="
            f"{summary['mse_mean']:.5f} "
            f"MAE="
            f"{summary['mae_mean']:.5f} "
            f"PSNR="
            f"{summary['psnr_mean']:.4f} "
            f"SSIM="
            f"{summary['ssim_mean']:.5f}"
        )

        print(
            f"wins vs hint: "
            f"MSE {mse_wins}/{n}, "
            f"MAE {mae_wins}/{n}, "
            f"SSIM {ssim_wins}/{n}"
        )

        print(
            f"MSE gain vs hint = "
            f"{summary['mse_gain_vs_hint']:+.5f}"
        )

    # ----------------------------------------------------
    # Save CSV
    # ----------------------------------------------------

    summary_csv = (
        out /
        "fewstep_summary.csv"
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
        "fewstep_per_sample.csv"
    )

    with sample_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                sample_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            sample_rows
        )

    print()
    print("=" * 130)
    print("C4 FEW-STEP VAL20 RANKING")
    print("=" * 130)

    print(
        f"{'schedule':>16} "
        f"{'NFE':>4} "
        f"{'MSE':>11} "
        f"{'MAE':>10} "
        f"{'PSNR':>9} "
        f"{'SSIM':>9} "
        f"{'MSEwin':>8} "
        f"{'MAEwin':>8} "
        f"{'SSIMwin':>9} "
        f"{'MSEgain':>11}"
    )

    for s in sorted(
        summaries,
        key=lambda x:
            x["mse_mean"],
    ):

        print(
            f"{s['schedule']:>16s} "
            f"{s['nfe']:4d} "
            f"{s['mse_mean']:11.4f} "
            f"{s['mae_mean']:10.4f} "
            f"{s['psnr_mean']:9.4f} "
            f"{s['ssim_mean']:9.5f} "
            f"{s['mse_wins_vs_hint']:5d}/{n:<2d} "
            f"{s['mae_wins_vs_hint']:5d}/{n:<2d} "
            f"{s['ssim_wins_vs_hint']:5d}/{n:<2d} "
            f"{s['mse_gain_vs_hint']:+11.4f}"
        )

    best = min(
        summaries,
        key=lambda x:
            x["mse_mean"],
    )

    print()
    print(
        "best few-step schedule =",
        best["schedule"],
    )

    print(
        "best NFE =",
        best["nfe"],
    )

    print(
        "best ts =",
        best["ts"],
    )

    print(
        "best MSE =",
        best["mse_mean"],
    )

    print(
        "hint MSE =",
        hint_mse,
    )

    print(
        "gain vs hint =",
        best["mse_gain_vs_hint"],
    )

    with open(
        out /
        "fewstep_summary.json",
        "w",
    ) as f:

        json.dump(
            {
                "hint_mse":
                    float(hint_mse),
                "hint_mae":
                    float(hint_mae),
                "hint_ssim":
                    float(hint_ssim),
                "results":
                    summaries,
                "best":
                    best,
            },
            f,
            indent=2,
        )

    print()
    print(
        "[PASS] C4 few-step "
        "VAL20 evaluation complete"
    )


if __name__ == "__main__":
    main()
