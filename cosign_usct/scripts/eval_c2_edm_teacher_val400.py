from pathlib import Path
import csv
import json
import re

import numpy as np
import torch

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
    "experiments/c2_edm_teacher_3600_c256_r2_e20k_v1"
)

VAL_FILE = (
    WORK /
    "data_3600_400_formal/val_gt_norm.npy"
)

OUT = (
    WORK /
    "metrics/c2_edm_teacher_3600_val400"
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Candidate checkpoints
# ============================================================

VALID_STEPS = {
    5000,
    10000,
    15000,
    20000,
}

VALID_EMAS = {
    "0.999",
    "0.9999",
    "0.9999432189950708",
}


def discover_candidates():

    candidates = {}

    pattern = re.compile(
        r"ema_(.+)_(\d+)\.pt$"
    )

    for path in sorted(RUN.glob("ema_*.pt")):

        m = pattern.match(path.name)

        if m is None:
            continue

        ema = m.group(1)
        step = int(m.group(2))

        if step not in VALID_STEPS:
            continue

        if ema not in VALID_EMAS:
            continue

        name = (
            f"ema_{ema}_"
            f"{step:06d}"
        )

        candidates[name] = path

    return candidates


# ============================================================
# Evaluation protocol
# ============================================================

BASE_SEED = 20260926

# Keep the same repeated-noise protocol as the old C2 evaluator.
NUM_REPEATS = 8

SIGMA_MIN = 0.002
SIGMA_MAX = 80.0

# CoSIGN / EDM LogNormalSampler
P_MEAN = -1.2
P_STD = 1.2


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
        "dropout": 0.1,

        "use_fp16": True,

        "weight_schedule": "karras",

        "sigma_min": SIGMA_MIN,
        "sigma_max": SIGMA_MAX,

        "loss_norm": "l2",
        "control": False,
    })

    return create_model_and_diffusion(**cfg)


def make_fixed_sigmas(n, repeats):

    rng = np.random.RandomState(
        BASE_SEED
    )

    z = rng.randn(
        n,
        repeats,
    )

    sigmas = np.exp(
        P_MEAN +
        P_STD * z
    ).astype(np.float32)

    sigmas = np.clip(
        sigmas,
        SIGMA_MIN,
        SIGMA_MAX,
    )

    return sigmas


@torch.no_grad()
def evaluate_candidate(
    name,
    ckpt,
    val,
    sigmas,
    device,
):

    print()
    print("=" * 90)
    print("candidate :", name)
    print("checkpoint:", ckpt)
    print("=" * 90)

    model, diffusion = make_model()

    state = torch.load(
        ckpt,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(device)

    if hasattr(
        model,
        "convert_to_fp16",
    ):
        model.convert_to_fp16()

    model.eval()

    weighted_losses = []
    xs_losses = []

    rows = []

    n = len(val)

    for i in range(n):

        x0 = torch.from_numpy(
            val[i:i+1, None]
        ).float().to(device)

        for r in range(NUM_REPEATS):

            sigma_value = float(
                sigmas[i, r]
            )

            sigma = torch.tensor(
                [sigma_value],
                dtype=torch.float32,
                device=device,
            )

            # Fixed paired noise:
            # exactly identical input noise for every checkpoint.
            noise_seed = (
                BASE_SEED +
                i * 1000 +
                r
            )

            gen = torch.Generator(
                device=device
            )

            gen.manual_seed(
                noise_seed
            )

            noise = torch.randn(
                x0.shape,
                generator=gen,
                device=device,
                dtype=x0.dtype,
            )

            terms = (
                diffusion.training_losses(
                    model,
                    x0,
                    sigma,
                    model_kwargs={},
                    noise=noise,
                )
            )

            weighted = float(
                terms["mse"]
                .mean()
                .item()
            )

            xs = float(
                terms["xs_mse"]
                .mean()
                .item()
            )

            if not np.isfinite(weighted):
                raise RuntimeError(
                    f"non-finite weighted mse: "
                    f"{name}, sample={i+1}, "
                    f"repeat={r}"
                )

            if not np.isfinite(xs):
                raise RuntimeError(
                    f"non-finite xs_mse: "
                    f"{name}, sample={i+1}, "
                    f"repeat={r}"
                )

            weighted_losses.append(
                weighted
            )

            xs_losses.append(
                xs
            )

            rows.append({
                "candidate": name,
                "sample_id": i + 1,
                "repeat": r,
                "noise_seed": noise_seed,
                "sigma": sigma_value,
                "weighted_mse": weighted,
                "xs_mse": xs,
            })

        if (i + 1) % 25 == 0:

            print(
                f"{name}: "
                f"{i+1:03d}/{n} "
                f"| weighted="
                f"{np.mean(weighted_losses):.8f} "
                f"| xs="
                f"{np.mean(xs_losses):.8f}"
            )

    summary = {

        "candidate":
            name,

        "checkpoint":
            str(ckpt),

        "n_val":
            int(n),

        "num_repeats":
            NUM_REPEATS,

        "num_forward_evals":
            int(n * NUM_REPEATS),

        "weighted_mse_mean":
            float(
                np.mean(
                    weighted_losses
                )
            ),

        "weighted_mse_std":
            float(
                np.std(
                    weighted_losses
                )
            ),

        "xs_mse_mean":
            float(
                np.mean(
                    xs_losses
                )
            ),

        "xs_mse_std":
            float(
                np.std(
                    xs_losses
                )
            ),

        "sigma_mean":
            float(
                sigmas.mean()
            ),

        "sigma_min":
            float(
                sigmas.min()
            ),

        "sigma_max":
            float(
                sigmas.max()
            ),
    }

    print()
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    del state
    del model
    del diffusion

    torch.cuda.empty_cache()

    return summary, rows


def main():

    torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device,
    )

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for "
            "formal VAL400 evaluation."
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
        val.shape ==
        (400, 256, 256)
    )

    assert np.isfinite(
        val
    ).all()

    candidates = (
        discover_candidates()
    )

    print()
    print(
        "Number of candidates =",
        len(candidates)
    )

    for name, path in candidates.items():

        print(
            name,
            "->",
            path.name
        )

    if len(candidates) != 12:

        print()
        print(
            "[WARNING] expected 12 "
            "EMA checkpoints, found",
            len(candidates)
        )

    if not candidates:

        raise RuntimeError(
            "No candidate checkpoints found."
        )

    sigmas = make_fixed_sigmas(
        len(val),
        NUM_REPEATS,
    )

    print()
    print(
        "sigma range =",
        float(sigmas.min()),
        float(sigmas.max())
    )

    print(
        "sigma mean =",
        float(sigmas.mean())
    )

    all_summaries = []
    all_rows = []

    for name, ckpt in (
        candidates.items()
    ):

        summary, rows = (
            evaluate_candidate(
                name,
                ckpt,
                val,
                sigmas,
                device,
            )
        )

        all_summaries.append(
            summary
        )

        all_rows.extend(
            rows
        )

    # ========================================================
    # Model selection
    #
    # Primary:
    # training-consistent Karras weighted MSE.
    #
    # Secondary:
    # clean-space xs_mse.
    # ========================================================

    all_summaries.sort(
        key=lambda x: (
            x[
                "weighted_mse_mean"
            ],
            x[
                "xs_mse_mean"
            ],
        )
    )

    for rank, item in enumerate(
        all_summaries,
        start=1,
    ):

        item["rank"] = rank

    best = all_summaries[0]

    result = {

        "protocol": {

            "train_set":
                "OpenBreastUS formal train3600",

            "selection_set":
                "formal val400",

            "test_set_used":
                False,

            "selection_metric":
                "fixed paired-noise "
                "Karras-weighted EDM MSE",

            "secondary_metric":
                "fixed paired-noise "
                "clean-space xs_mse",

            "num_repeats":
                NUM_REPEATS,

            "base_seed":
                BASE_SEED,

            "lognormal_p_mean":
                P_MEAN,

            "lognormal_p_std":
                P_STD,

            "sigma_min":
                SIGMA_MIN,

            "sigma_max":
                SIGMA_MAX,

            "note":
                "All candidate checkpoints "
                "receive identical validation "
                "images, sigma values, and "
                "noise realizations. "
                "Test400 is untouched.",
        },

        "ranking":
            all_summaries,

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
            "weighted_mse_mean",
            "weighted_mse_std",
            "xs_mse_mean",
            "xs_mse_std",
            "n_val",
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
            all_summaries
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
            "repeat",
            "noise_seed",
            "sigma",
            "weighted_mse",
            "xs_mse",
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
    print("FINAL C2 VAL400 RANKING")
    print("=" * 90)

    for x in all_summaries:

        print(
            f'#{x["rank"]:02d} '
            f'{x["candidate"]:<32s} '
            f'weighted='
            f'{x["weighted_mse_mean"]:.8f} '
            f'xs='
            f'{x["xs_mse_mean"]:.8f}'
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
        "[PASS] C2 EDM teacher "
        "formal VAL400 selection complete."
    )


if __name__ == "__main__":
    main()
