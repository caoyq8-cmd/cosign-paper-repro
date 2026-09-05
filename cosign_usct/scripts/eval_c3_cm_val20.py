from pathlib import Path
import csv
import json
import math

import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SSIM = True
except Exception:
    HAS_SSIM = False

from cc.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


# ============================================================
# Paths
# ============================================================

WORK = Path(
    "/home/featurize/work/USCT_repro/"
    "paper_reproduction/cosign_usct"
)

RUN = WORK / "experiments/c3_cm_c256_e50k_v1"

VAL_FILE = WORK / "data/val20_gt_norm.npy"

OUT = WORK / "metrics/c3_cm_eval"
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "cm25k_ema09999":
        RUN / "ema_0.9999_025000.pt",

    "cm50k_ema09999":
        RUN / "ema_0.9999_050000.pt",
}


# ============================================================
# Frozen protocol
# ============================================================

BASE_SEED = 20260903

SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0

NUM_SCALES = 40

# Representative points across the same 40-scale Karras grid
# used in consistency distillation.
SCALE_INDICES = [
    0, 6, 12, 18, 24, 30, 36, 39
]

NUM_REPEATS = 2

SPEED_MIN = 1400.0
SPEED_MAX = 1605.0
SPEED_RANGE = SPEED_MAX - SPEED_MIN


def norm_to_speed(x):
    return SPEED_MIN + (x + 1.0) * 0.5 * SPEED_RANGE


def karras_sigma(index):
    a = SIGMA_MAX ** (1.0 / RHO)
    b = SIGMA_MIN ** (1.0 / RHO)

    sigma = (
        a
        + index / (NUM_SCALES - 1)
        * (b - a)
    ) ** RHO

    return float(sigma)


SIGMAS = [
    karras_sigma(i)
    for i in SCALE_INDICES
]


def make_model():
    cfg = model_and_diffusion_defaults()

    cfg.update({
        "image_size": 256,
        "in_channels": 1,
        "class_cond": False,

        "num_channels": 256,
        "num_res_blocks": 2,

        "attention_resolutions": "32,16,8",
        "num_head_channels": 64,

        "use_scale_shift_norm": False,
        "resblock_updown": True,

        # Student CM configuration
        "dropout": 0.0,
        "use_fp16": True,

        "weight_schedule": "uniform",
        "loss_norm": "l2",
        "loss_type": "consistency",

        "sigma_min": SIGMA_MIN,
        "sigma_max": SIGMA_MAX,

        "control": False,
    })

    # Critical:
    # consistency model uses boundary-condition scalings.
    model, diffusion = create_model_and_diffusion(
        **cfg,
        distillation=True,
    )

    return model, diffusion


def calc_metrics(pred_norm, gt_norm):

    # Raw normalized-space error.
    raw_diff = pred_norm - gt_norm

    norm_mse = float(
        np.mean(raw_diff ** 2)
    )

    norm_mae = float(
        np.mean(np.abs(raw_diff))
    )

    # Sampling ultimately operates in [-1, 1].
    pred_clip = np.clip(
        pred_norm, -1.0, 1.0
    )

    gt_clip = np.clip(
        gt_norm, -1.0, 1.0
    )

    pred_speed = norm_to_speed(pred_clip)
    gt_speed = norm_to_speed(gt_clip)

    diff_speed = pred_speed - gt_speed

    speed_mse = float(
        np.mean(diff_speed ** 2)
    )

    speed_mae = float(
        np.mean(np.abs(diff_speed))
    )

    speed_rmse = math.sqrt(speed_mse)

    if speed_rmse > 0:
        psnr = 20.0 * math.log10(
            SPEED_RANGE / speed_rmse
        )
    else:
        psnr = float("inf")

    if HAS_SSIM:
        ssim = float(
            skimage_ssim(
                gt_speed,
                pred_speed,
                data_range=SPEED_RANGE,
            )
        )
    else:
        ssim = float("nan")

    return {
        "norm_mse": norm_mse,
        "norm_mae": norm_mae,

        "speed_mse": speed_mse,
        "speed_mae": speed_mae,

        "psnr": psnr,
        "ssim": ssim,
    }


@torch.no_grad()
def evaluate_candidate(
    name,
    checkpoint,
    val,
    device,
):

    print()
    print("=" * 80)
    print("candidate :", name)
    print("checkpoint:", checkpoint)
    print("=" * 80)

    model, diffusion = make_model()

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(state)

    del state

    model.to(device)
    model.convert_to_fp16()
    model.eval()

    rows = []

    for sample_idx in range(len(val)):

        gt_np = val[sample_idx]

        x0 = torch.from_numpy(
            gt_np[None, None, :, :]
        ).float().to(device)

        for scale_pos, (
            scale_index,
            sigma_value,
        ) in enumerate(
            zip(
                SCALE_INDICES,
                SIGMAS,
            )
        ):

            for repeat in range(
                NUM_REPEATS
            ):

                noise_seed = (
                    BASE_SEED
                    + sample_idx * 10000
                    + scale_pos * 100
                    + repeat
                )

                generator = torch.Generator(
                    device=device
                )

                generator.manual_seed(
                    noise_seed
                )

                noise = torch.randn(
                    x0.shape,
                    generator=generator,
                    device=device,
                    dtype=x0.dtype,
                )

                sigma = torch.tensor(
                    [sigma_value],
                    device=device,
                    dtype=torch.float32,
                )

                x_t = (
                    x0
                    + sigma.reshape(1,1,1,1)
                    * noise
                )

                _, denoised = diffusion.denoise(
                    model,
                    x_t,
                    sigma,
                )

                pred_np = (
                    denoised[0, 0]
                    .float()
                    .cpu()
                    .numpy()
                )

                metrics = calc_metrics(
                    pred_np,
                    gt_np,
                )

                row = {
                    "candidate": name,
                    "sample_id":
                        sample_idx + 1,

                    "scale_index":
                        scale_index,

                    "sigma":
                        sigma_value,

                    "repeat":
                        repeat,

                    "noise_seed":
                        noise_seed,
                }

                row.update(metrics)

                rows.append(row)

        if (sample_idx + 1) % 5 == 0:
            print(
                f"{name}: "
                f"{sample_idx+1}/"
                f"{len(val)} complete"
            )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    def mean(key):
        vals = np.asarray(
            [r[key] for r in rows],
            dtype=np.float64,
        )

        return float(
            np.nanmean(vals)
        )

    def std(key):
        vals = np.asarray(
            [r[key] for r in rows],
            dtype=np.float64,
        )

        return float(
            np.nanstd(vals)
        )

    summary = {
        "candidate": name,
        "checkpoint":
            str(checkpoint),

        "n_val":
            len(val),

        "num_scales":
            len(SIGMAS),

        "num_repeats":
            NUM_REPEATS,

        "num_forward_evals":
            len(rows),

        "norm_mse_mean":
            mean("norm_mse"),

        "norm_mse_std":
            std("norm_mse"),

        "norm_mae_mean":
            mean("norm_mae"),

        "speed_mse_mean":
            mean("speed_mse"),

        "speed_mae_mean":
            mean("speed_mae"),

        "psnr_mean":
            mean("psnr"),

        "ssim_mean":
            mean("ssim"),

        "ssim_available":
            HAS_SSIM,
    }

    print()
    print(
        json.dumps(
            summary,
            indent=2
        )
    )

    del model
    del diffusion

    torch.cuda.empty_cache()

    return summary, rows


def main():

    device = torch.device(
        "cuda:0"
    )

    val = np.load(
        VAL_FILE
    ).astype(np.float32)

    assert val.shape == (
        20, 256, 256
    )

    assert np.isfinite(
        val
    ).all()

    print("VAL shape =", val.shape)
    print(
        "VAL range =",
        float(val.min()),
        float(val.max())
    )

    print()
    print("Karras validation scales:")

    for idx, sigma in zip(
        SCALE_INDICES,
        SIGMAS,
    ):
        print(
            f"index={idx:2d} "
            f"sigma={sigma:.8f}"
        )

    summaries = []
    all_rows = []

    for name, checkpoint in (
        CANDIDATES.items()
    ):

        if not checkpoint.exists():
            raise FileNotFoundError(
                checkpoint
            )

        summary, rows = (
            evaluate_candidate(
                name,
                checkpoint,
                val,
                device,
            )
        )

        summaries.append(
            summary
        )

        all_rows.extend(
            rows
        )

    # Primary selection:
    # raw normalized x0 denoising MSE.
    summaries.sort(
        key=lambda x:
            x["norm_mse_mean"]
    )

    for rank, item in enumerate(
        summaries,
        start=1
    ):
        item["rank"] = rank

    best = summaries[0]

    with open(
        OUT / "summary.json",
        "w"
    ) as f:
        json.dump(
            {
                "protocol": {
                    "dataset":
                        "VAL20",

                    "scale_indices":
                        SCALE_INDICES,

                    "sigmas":
                        SIGMAS,

                    "num_repeats":
                        NUM_REPEATS,

                    "base_seed":
                        BASE_SEED,

                    "selection_metric":
                        "raw normalized clean-space MSE",

                    "note":
                        "All checkpoints use identical "
                        "GT, sigma and noise seeds."
                },

                "ranking":
                    summaries,

                "best":
                    best,
            },
            f,
            indent=2,
        )

    with open(
        OUT / "summary.csv",
        "w",
        newline=""
    ) as f:

        fields = [
            "rank",
            "candidate",

            "norm_mse_mean",
            "norm_mse_std",
            "norm_mae_mean",

            "speed_mse_mean",
            "speed_mae_mean",

            "psnr_mean",
            "ssim_mean",

            "n_val",
            "num_scales",
            "num_repeats",
            "num_forward_evals",

            "checkpoint",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    with open(
        OUT / "per_eval.csv",
        "w",
        newline=""
    ) as f:

        fields = [
            "candidate",
            "sample_id",
            "scale_index",
            "sigma",
            "repeat",
            "noise_seed",

            "norm_mse",
            "norm_mae",

            "speed_mse",
            "speed_mae",

            "psnr",
            "ssim",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )

    print()
    print("=" * 80)
    print("FINAL RANKING")
    print("=" * 80)

    for item in summaries:
        print(
            f'#{item["rank"]} '
            f'{item["candidate"]} | '
            f'norm_mse='
            f'{item["norm_mse_mean"]:.8f} | '
            f'speed_mse='
            f'{item["speed_mse_mean"]:.4f} | '
            f'MAE='
            f'{item["speed_mae_mean"]:.4f} | '
            f'PSNR='
            f'{item["psnr_mean"]:.4f} | '
            f'SSIM='
            f'{item["ssim_mean"]:.6f}'
        )

    print()
    print(
        "BEST =",
        best["candidate"]
    )

    print(
        "BEST CHECKPOINT =",
        best["checkpoint"]
    )

    print()
    print(
        "[PASS] C3 CM VAL20 "
        "checkpoint selection complete."
    )


if __name__ == "__main__":
    main()
