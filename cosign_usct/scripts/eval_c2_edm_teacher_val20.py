from pathlib import Path
import csv
import json
import math

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

RUN = WORK / "experiments/c2_edm_teacher_c256_r2_e20k_v1"

VAL_FILE = WORK / "data/val20_gt_norm.npy"

OUT = WORK / "metrics/c2_edm_teacher_eval"
OUT.mkdir(parents=True, exist_ok=True)


CANDIDATES = {
    "ema09999_10k":
        RUN / "ema_0.9999_010000.pt",

    "ema0999943_10k":
        RUN / "ema_0.9999432189950708_010000.pt",

    "ema09999_20k":
        RUN / "ema_0.9999_020000.pt",

    "ema0999943_20k":
        RUN / "ema_0.9999432189950708_020000.pt",
}


# ============================================================
# Evaluation protocol
# ============================================================

BASE_SEED = 20260903

# Fixed repeated noise realizations per validation image.
NUM_REPEATS = 8

SIGMA_MIN = 0.002
SIGMA_MAX = 80.0

# Match official CoSIGN LogNormalSampler:
# log sigma ~ N(-1.2, 1.2^2)
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
    rng = np.random.RandomState(BASE_SEED)

    z = rng.randn(n, repeats)

    sigmas = np.exp(
        P_MEAN + P_STD * z
    ).astype(np.float32)

    # Keep evaluation inside model-supported range.
    sigmas = np.clip(
        sigmas,
        SIGMA_MIN,
        SIGMA_MAX
    )

    return sigmas


@torch.no_grad()
def evaluate_candidate(name, ckpt, val, sigmas, device):

    print()
    print("=" * 80)
    print("candidate:", name)
    print("checkpoint:", ckpt)
    print("=" * 80)

    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    model, diffusion = make_model()

    state = torch.load(
        ckpt,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(state)

    model.to(device)
    model.convert_to_fp16()
    model.eval()

    rows = []

    weighted_losses = []
    xs_losses = []

    n = val.shape[0]

    for i in range(n):

        x0 = torch.from_numpy(
            val[i:i+1, None, :, :]
        ).float().to(device)

        for r in range(NUM_REPEATS):

            sigma_value = float(sigmas[i, r])

            sigma = torch.tensor(
                [sigma_value],
                dtype=torch.float32,
                device=device,
            )

            # Deterministic noise for paired checkpoint comparison.
            noise_seed = (
                BASE_SEED
                + i * 1000
                + r
            )

            gen = torch.Generator(
                device=device
            )
            gen.manual_seed(noise_seed)

            noise = torch.randn(
                x0.shape,
                generator=gen,
                device=device,
                dtype=x0.dtype,
            )

            terms = diffusion.training_losses(
                model,
                x0,
                sigma,
                model_kwargs={},
                noise=noise,
            )

            weighted = float(
                terms["mse"].mean().item()
            )

            xs = float(
                terms["xs_mse"].mean().item()
            )

            weighted_losses.append(weighted)
            xs_losses.append(xs)

            rows.append({
                "candidate": name,
                "sample_id": i + 1,
                "repeat": r,
                "noise_seed": noise_seed,
                "sigma": sigma_value,
                "weighted_mse": weighted,
                "xs_mse": xs,
            })

        if (i + 1) % 5 == 0:
            print(
                f"{name}: "
                f"{i+1}/{n} samples evaluated"
            )

    summary = {
        "candidate": name,
        "checkpoint": str(ckpt),

        "n_val": int(n),
        "num_repeats": int(NUM_REPEATS),
        "num_forward_evals": int(
            n * NUM_REPEATS
        ),

        "weighted_mse_mean":
            float(np.mean(weighted_losses)),

        "weighted_mse_std":
            float(np.std(weighted_losses)),

        "xs_mse_mean":
            float(np.mean(xs_losses)),

        "xs_mse_std":
            float(np.std(xs_losses)),

        "sigma_mean":
            float(sigmas.mean()),

        "sigma_min":
            float(sigmas.min()),

        "sigma_max":
            float(sigmas.max()),
    }

    print(json.dumps(summary, indent=2))

    del model
    del diffusion
    torch.cuda.empty_cache()

    return summary, rows


def main():

    device = torch.device("cuda:0")

    val = np.load(VAL_FILE).astype(
        np.float32
    )

    print("VAL shape =", val.shape)
    print("VAL dtype =", val.dtype)
    print(
        "VAL range =",
        float(val.min()),
        float(val.max())
    )

    assert val.shape == (20, 256, 256)
    assert np.isfinite(val).all()

    sigmas = make_fixed_sigmas(
        len(val),
        NUM_REPEATS,
    )

    print(
        "sigma range =",
        float(sigmas.min()),
        float(sigmas.max())
    )

    all_summaries = []
    all_rows = []

    for name, ckpt in CANDIDATES.items():

        if not ckpt.exists():
            print(
                "[SKIP missing]",
                name,
                ckpt
            )
            continue

        summary, rows = evaluate_candidate(
            name,
            ckpt,
            val,
            sigmas,
            device,
        )

        all_summaries.append(summary)
        all_rows.extend(rows)

    if not all_summaries:
        raise RuntimeError(
            "No candidate checkpoints found."
        )

    # --------------------------------------------------------
    # Primary selection:
    # lowest fixed-protocol weighted EDM validation loss
    #
    # Secondary:
    # lowest clean-space denoising xs_mse
    # --------------------------------------------------------

    all_summaries.sort(
        key=lambda x: (
            x["weighted_mse_mean"],
            x["xs_mse_mean"],
        )
    )

    for rank, item in enumerate(
        all_summaries,
        start=1
    ):
        item["rank"] = rank

    best = all_summaries[0]

    with open(
        OUT / "summary.json",
        "w"
    ) as f:
        json.dump(
            {
                "protocol": {
                    "dataset":
                        "VAL20 test_1..test_20",

                    "selection_metric":
                        "fixed-noise weighted EDM validation MSE",

                    "secondary_metric":
                        "fixed-noise clean-space xs_mse",

                    "lognormal_p_mean":
                        P_MEAN,

                    "lognormal_p_std":
                        P_STD,

                    "num_repeats":
                        NUM_REPEATS,

                    "base_seed":
                        BASE_SEED,

                    "note":
                        "unconditional EDM samples are not paired "
                        "with VAL20 GT; candidate selection therefore "
                        "uses denoising validation, not random-sample "
                        "image-to-GT MSE."
                },

                "ranking":
                    all_summaries,

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
        writer.writerows(all_summaries)

    with open(
        OUT / "per_eval.csv",
        "w",
        newline=""
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
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("=" * 80)
    print("FINAL RANKING")
    print("=" * 80)

    for x in all_summaries:
        print(
            f'#{x["rank"]} '
            f'{x["candidate"]}: '
            f'weighted_mse='
            f'{x["weighted_mse_mean"]:.8f}, '
            f'xs_mse='
            f'{x["xs_mse_mean"]:.8f}'
        )

    print()
    print("BEST =", best["candidate"])
    print(
        "BEST CHECKPOINT =",
        best["checkpoint"]
    )

    print()
    print(
        "[PASS] C2 EDM teacher "
        "VAL20 selection complete."
    )


if __name__ == "__main__":
    main()
