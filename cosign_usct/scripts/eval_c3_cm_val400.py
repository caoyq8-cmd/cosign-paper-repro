from pathlib import Path
import csv
import json
import math
import re

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

RUN = (
    WORK /
    "experiments/c3_cm_3600_cd12k_v1"
)

VAL_FILE = (
    WORK /
    "data_3600_400_formal/val_gt_norm.npy"
)

OUT = (
    WORK /
    "metrics/c3_cm_3600_val400"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Candidate discovery
# ============================================================

VALID_STEPS = {
    3000,
    6000,
    9000,
    12000,
}

VALID_EMAS = {
    "0.9999",
    "0.99994",
    "0.9999432189950708",
}


def discover_candidates():

    pattern = re.compile(
        r"ema_(.+)_(\d+)\.pt$"
    )

    candidates = {}

    for path in sorted(
        RUN.glob("ema_*.pt")
    ):

        m = pattern.match(
            path.name
        )

        if m is None:
            continue

        ema = m.group(1)
        step = int(m.group(2))

        if step not in VALID_STEPS:
            continue

        if ema not in VALID_EMAS:
            continue

        name = (
            f"cm_ema_{ema}_"
            f"{step:06d}"
        )

        candidates[name] = path

    return candidates


# ============================================================
# Frozen validation protocol
# ============================================================

BASE_SEED = 20260927

SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0

NUM_SCALES = 40

# Representative locations on the exact
# 40-scale Karras grid used in training.
SCALE_INDICES = [
    0,
    6,
    12,
    18,
    24,
    30,
    36,
    39,
]

# VAL400 is large enough that one paired
# deterministic noise realization per scale
# is sufficient for checkpoint selection.
NUM_REPEATS = 1


# ------------------------------------------------------------
# Physical speed convention used in current USCT experiments.
# Primary checkpoint selection does NOT depend on this range;
# normalized-space MSE is the primary metric.
# ------------------------------------------------------------

SPEED_MIN = 1400.0
SPEED_MAX = 1605.0
SPEED_RANGE = SPEED_MAX - SPEED_MIN


def norm_to_speed(x):

    return (
        SPEED_MIN
        + (x + 1.0)
        * 0.5
        * SPEED_RANGE
    )


def karras_sigma(index):

    a = SIGMA_MAX ** (
        1.0 / RHO
    )

    b = SIGMA_MIN ** (
        1.0 / RHO
    )

    sigma = (
        a
        + index
        / (NUM_SCALES - 1)
        * (b - a)
    ) ** RHO

    return float(sigma)


SIGMAS = [
    karras_sigma(i)
    for i in SCALE_INDICES
]


# ============================================================
# CM construction
# ============================================================

def make_model():

    cfg = (
        model_and_diffusion_defaults()
    )

    cfg.update({

        "image_size": 256,
        "in_channels": 1,

        "class_cond": False,

        "num_channels": 256,
        "num_res_blocks": 2,

        "attention_resolutions":
            "32,16,8",

        "num_head_channels": 64,

        "use_scale_shift_norm":
            False,

        "resblock_updown": True,

        # CM student architecture
        "dropout": 0.0,

        "use_fp16": True,

        # The following loss choice does not
        # alter inference/denoise behavior.
        "loss_norm": "l2",

        "loss_type":
            "consistency",

        "weight_schedule":
            "uniform",

        "sigma_min":
            SIGMA_MIN,

        "sigma_max":
            SIGMA_MAX,

        "control": False,
    })

    model, diffusion = (
        create_model_and_diffusion(
            **cfg,
            distillation=True,
        )
    )

    return model, diffusion


# ============================================================
# Metrics
# ============================================================

def calc_metrics(
    pred_norm,
    gt_norm,
):

    raw_diff = (
        pred_norm - gt_norm
    )

    norm_mse = float(
        np.mean(
            raw_diff ** 2
        )
    )

    norm_mae = float(
        np.mean(
            np.abs(raw_diff)
        )
    )

    pred_clip = np.clip(
        pred_norm,
        -1.0,
        1.0,
    )

    gt_clip = np.clip(
        gt_norm,
        -1.0,
        1.0,
    )

    pred_speed = norm_to_speed(
        pred_clip
    )

    gt_speed = norm_to_speed(
        gt_clip
    )

    diff = (
        pred_speed
        - gt_speed
    )

    speed_mse = float(
        np.mean(diff ** 2)
    )

    speed_mae = float(
        np.mean(
            np.abs(diff)
        )
    )

    speed_rmse = math.sqrt(
        speed_mse
    )

    if speed_rmse > 0:

        psnr = (
            20.0
            * math.log10(
                SPEED_RANGE
                / speed_rmse
            )
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
        "norm_mse":
            norm_mse,

        "norm_mae":
            norm_mae,

        "speed_mse":
            speed_mse,

        "speed_mae":
            speed_mae,

        "psnr":
            psnr,

        "ssim":
            ssim,
    }


# ============================================================
# Candidate evaluation
# ============================================================

@torch.no_grad()
def evaluate_candidate(
    name,
    checkpoint,
    val,
    device,
):

    print()
    print("=" * 90)
    print(
        "candidate :",
        name
    )
    print(
        "checkpoint:",
        checkpoint
    )
    print("=" * 90)

    model, diffusion = (
        make_model()
    )

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    del state

    model.to(device)

    if hasattr(
        model,
        "convert_to_fp16"
    ):
        model.convert_to_fp16()

    model.eval()

    rows = []

    n = len(val)

    for sample_idx in range(n):

        gt_np = val[
            sample_idx
        ]

        x0 = torch.from_numpy(
            gt_np[
                None,
                None,
                :,
                :
            ]
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
                    + sample_idx
                    * 10000
                    + scale_pos
                    * 100
                    + repeat
                )

                generator = (
                    torch.Generator(
                        device=device
                    )
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
                    + sigma.reshape(
                        1, 1, 1, 1
                    )
                    * noise
                )

                _, denoised = (
                    diffusion.denoise(
                        model,
                        x_t,
                        sigma,
                    )
                )

                pred_np = (
                    denoised[
                        0, 0
                    ]
                    .float()
                    .cpu()
                    .numpy()
                )

                metrics = (
                    calc_metrics(
                        pred_np,
                        gt_np,
                    )
                )

                for key, value in (
                    metrics.items()
                ):

                    if not np.isfinite(
                        value
                    ):
                        if (
                            key == "ssim"
                            and not HAS_SSIM
                        ):
                            continue

                        raise RuntimeError(
                            "non-finite metric: "
                            f"{name}, "
                            f"sample="
                            f"{sample_idx+1}, "
                            f"scale="
                            f"{scale_index}, "
                            f"{key}={value}"
                        )

                row = {
                    "candidate":
                        name,

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

                row.update(
                    metrics
                )

                rows.append(
                    row
                )

        if (
            sample_idx + 1
        ) % 25 == 0:

            current_mse = float(
                np.mean(
                    [
                        r["norm_mse"]
                        for r
                        in rows
                    ]
                )
            )

            print(
                f"{name}: "
                f"{sample_idx+1:03d}/"
                f"{n} "
                f"| running norm_mse="
                f"{current_mse:.8f}"
            )

    def mean(key):

        vals = np.asarray(
            [
                r[key]
                for r in rows
            ],
            dtype=np.float64,
        )

        return float(
            np.nanmean(vals)
        )

    def std(key):

        vals = np.asarray(
            [
                r[key]
                for r in rows
            ],
            dtype=np.float64,
        )

        return float(
            np.nanstd(vals)
        )

    summary = {

        "candidate":
            name,

        "checkpoint":
            str(checkpoint),

        "n_val":
            n,

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
            indent=2,
        )
    )

    del model
    del diffusion

    torch.cuda.empty_cache()

    return summary, rows


# ============================================================
# Main
# ============================================================

def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda:0"
    )

    val = np.load(
        VAL_FILE
    ).astype(
        np.float32
    )

    print(
        "VAL shape =",
        val.shape
    )

    print(
        "VAL range =",
        float(val.min()),
        float(val.max())
    )

    assert (
        val.shape
        == (400, 256, 256)
    )

    assert np.isfinite(
        val
    ).all()

    print()
    print(
        "Karras validation scales:"
    )

    for idx, sigma in zip(
        SCALE_INDICES,
        SIGMAS,
    ):

        print(
            f"index={idx:2d} "
            f"sigma={sigma:.8f}"
        )

    candidates = (
        discover_candidates()
    )

    print()
    print(
        "Number of candidates =",
        len(candidates)
    )

    for name, path in (
        candidates.items()
    ):

        print(
            name,
            "->",
            path.name
        )

    if len(candidates) != 12:

        print()
        print(
            "[WARNING] expected "
            "12 C3 EMA checkpoints, "
            "found",
            len(candidates)
        )

    if not candidates:

        raise RuntimeError(
            "No C3 EMA candidates found."
        )

    summaries = []
    all_rows = []

    for name, checkpoint in (
        candidates.items()
    ):

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

    # --------------------------------------------------------
    # Primary selection:
    # clean-space normalized MSE.
    #
    # Secondary:
    # normalized MAE.
    # --------------------------------------------------------

    summaries.sort(
        key=lambda x: (
            x[
                "norm_mse_mean"
            ],
            x[
                "norm_mae_mean"
            ],
        )
    )

    for rank, item in enumerate(
        summaries,
        start=1,
    ):

        item["rank"] = rank

    best = summaries[0]

    result = {

        "protocol": {

            "train_set":
                "OpenBreastUS "
                "formal train3600",

            "selection_set":
                "formal val400",

            "test_set_used":
                False,

            "num_scales":
                len(SIGMAS),

            "scale_indices":
                SCALE_INDICES,

            "sigmas":
                SIGMAS,

            "num_repeats":
                NUM_REPEATS,

            "base_seed":
                BASE_SEED,

            "selection_metric":
                "fixed-noise "
                "normalized clean-space "
                "denoising MSE",

            "secondary_metric":
                "normalized clean-space MAE",

            "note":
                "All checkpoints receive "
                "identical validation "
                "images, Karras sigma "
                "values, and noise seeds. "
                "Test400 is untouched.",
        },

        "ranking":
            summaries,

        "best":
            best,
    }

    with open(
        OUT /
        "summary.json",
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    with open(
        OUT /
        "summary.csv",
        "w",
        newline="",
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
        OUT /
        "per_eval.csv",
        "w",
        newline="",
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
    print("=" * 90)
    print(
        "FINAL C3 VAL400 RANKING"
    )
    print("=" * 90)

    for item in summaries:

        print(
            f'#{item["rank"]:02d} '
            f'{item["candidate"]:<36s} '
            f'norm_mse='
            f'{item["norm_mse_mean"]:.8f} '
            f'norm_mae='
            f'{item["norm_mae_mean"]:.8f} '
            f'PSNR='
            f'{item["psnr_mean"]:.4f} '
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
        "[PASS] C3 CM formal "
        "VAL400 selection complete."
    )


if __name__ == "__main__":
    main()
