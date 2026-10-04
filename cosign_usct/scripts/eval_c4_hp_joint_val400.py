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


def create_models(backbone, device):

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

    state = th.load(
        backbone,
        map_location="cpu",
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    del state

    model.to(device)
    control_net.to(device)

    model.convert_to_fp16()
    control_net.convert_to_fp16()

    model.eval()
    control_net.eval()

    return control_net, model, diffusion


@th.no_grad()
def get_prediction(
    hint_np,
    noise_np,
    sigma_value,
    control_net,
    model,
    diffusion,
    device,
    batch_size,
):

    preds = []

    n = len(hint_np)

    for start in range(
        0,
        n,
        batch_size,
    ):

        end = min(
            start + batch_size,
            n,
        )

        hint = th.from_numpy(
            hint_np[
                start:end,
                None
            ]
        ).float().to(device)

        noise = th.from_numpy(
            noise_np[
                start:end,
                None
            ]
        ).float().to(device)

        sigma = th.full(
            (end - start,),
            float(sigma_value),
            dtype=th.float32,
            device=device,
        )

        # --------------------------------------------------
        # Hint-preserving noisy initialization
        # --------------------------------------------------

        x_t = (
            hint
            + sigma[
                :, None, None, None
            ]
            * noise
        )

        _, pred = diffusion.recon(
            model,
            control_net,
            x_t,
            hint,
            sigma,
        )

        preds.append(
            pred[:, 0]
            .float()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        preds,
        axis=0,
    ).astype(np.float32)


def calc_per_sample(
    pred,
    gt,
):

    diff_norm = (
        pred - gt
    )

    # Conversion to m/s can be done
    # analytically because normalization
    # is affine.
    mse = (
        np.mean(
            diff_norm ** 2,
            axis=(1, 2),
        )
        * SCALE ** 2
    )

    mae = (
        np.mean(
            np.abs(diff_norm),
            axis=(1, 2),
        )
        * SCALE
    )

    psnr = (
        10.0
        * np.log10(
            DATA_RANGE ** 2
            / np.maximum(
                mse,
                1e-12,
            )
        )
    )

    return (
        mse.astype(np.float64),
        mae.astype(np.float64),
        psnr.astype(np.float64),
    )


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

    run = Path(args.run)

    gt = np.load(
        args.gt
    ).astype(np.float32)

    hint = np.load(
        args.hint
    ).astype(np.float32)

    if gt.shape != (
        400,
        256,
        256,
    ):
        raise RuntimeError(
            f"Expected VAL400, got "
            f"{gt.shape}"
        )

    if hint.shape != gt.shape:
        raise RuntimeError(
            f"GT/hint mismatch: "
            f"{gt.shape} vs "
            f"{hint.shape}"
        )

    assert np.isfinite(gt).all()
    assert np.isfinite(hint).all()

    steps = [
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
    ]

    sigmas = [
        11.71851402,
        3.51969883,
        0.82294137,
    ]

    alphas = [
        0.0,
        0.03,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    ]

    # --------------------------------------------------
    # Same noise for every checkpoint / sigma.
    # This gives paired comparisons.
    # --------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    noise = rng.standard_normal(
        gt.shape
    ).astype(np.float32)

    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------

    (
        baseline_mse,
        baseline_mae,
        baseline_psnr,
    ) = calc_per_sample(
        hint,
        gt,
    )

    baseline = {
        "mse_mean":
            float(
                baseline_mse.mean()
            ),

        "mse_std":
            float(
                baseline_mse.std()
            ),

        "mae_mean":
            float(
                baseline_mae.mean()
            ),

        "psnr_mean":
            float(
                baseline_psnr.mean()
            ),
    }

    print()
    print("=" * 100)
    print(
        "INVERSIONNET VAL400 BASELINE"
    )
    print("=" * 100)
    print(
        json.dumps(
            baseline,
            indent=2,
        )
    )

    device = th.device(
        "cuda:0"
    )

    (
        control_net,
        model,
        diffusion,
    ) = create_models(
        args.backbone,
        device,
    )

    rows = []

    best_row = None
    best_raw = None
    best_refined = None
    best_per_sample = None

    # --------------------------------------------------
    # Joint checkpoint × sigma × alpha sweep
    # --------------------------------------------------

    for step in steps:

        ckpt = (
            run
            / f"model{step:06d}.pt"
        )

        if not ckpt.exists():
            print(
                "[WARN] missing:",
                ckpt,
            )
            continue

        print()
        print("=" * 110)
        print(
            "CHECKPOINT STEP =",
            step,
        )
        print(
            "checkpoint =",
            ckpt,
        )
        print("=" * 110)

        state = th.load(
            ckpt,
            map_location="cpu",
        )

        control_net.load_state_dict(
            state,
            strict=True,
        )

        del state

        control_net.eval()

        for sigma_value in sigmas:

            print(
                f"\nforward: "
                f"step={step} "
                f"sigma={sigma_value:.8f}"
            )

            raw_pred = get_prediction(
                hint_np=hint,
                noise_np=noise,
                sigma_value=sigma_value,
                control_net=control_net,
                model=model,
                diffusion=diffusion,
                device=device,
                batch_size=args.batch_size,
            )

            if not np.isfinite(
                raw_pred
            ).all():
                raise RuntimeError(
                    f"non-finite prediction "
                    f"step={step} "
                    f"sigma={sigma_value}"
                )

            residual = (
                raw_pred - hint
            )

            for alpha in alphas:

                refined = (
                    hint
                    + alpha
                    * residual
                )

                refined = np.clip(
                    refined,
                    -1.0,
                    1.0,
                )

                (
                    mse_v,
                    mae_v,
                    psnr_v,
                ) = calc_per_sample(
                    refined,
                    gt,
                )

                mse_wins = int(
                    np.sum(
                        mse_v
                        < baseline_mse
                    )
                )

                mae_wins = int(
                    np.sum(
                        mae_v
                        < baseline_mae
                    )
                )

                row = {
                    "step":
                        step,

                    "sigma":
                        sigma_value,

                    "alpha":
                        alpha,

                    "mse_mean":
                        float(
                            mse_v.mean()
                        ),

                    "mse_std":
                        float(
                            mse_v.std()
                        ),

                    "mae_mean":
                        float(
                            mae_v.mean()
                        ),

                    "psnr_mean":
                        float(
                            psnr_v.mean()
                        ),

                    "mse_wins":
                        mse_wins,

                    "mae_wins":
                        mae_wins,

                    "mse_gain_vs_hint":
                        float(
                            baseline[
                                "mse_mean"
                            ]
                            -
                            mse_v.mean()
                        ),

                    "mse_gain_pct":
                        float(
                            100.0
                            * (
                                baseline[
                                    "mse_mean"
                                ]
                                -
                                mse_v.mean()
                            )
                            /
                            baseline[
                                "mse_mean"
                            ]
                        ),
                }

                rows.append(row)

                print(
                    f"step={step:4d} "
                    f"sigma="
                    f"{sigma_value:11.8f} "
                    f"alpha={alpha:5.3f} | "
                    f"MSE="
                    f"{row['mse_mean']:9.4f} "
                    f"MAE="
                    f"{row['mae_mean']:7.4f} "
                    f"MSEwin="
                    f"{mse_wins:3d}/400 "
                    f"gain="
                    f"{row['mse_gain_vs_hint']:+.4f}"
                )

                # Primary selection = mean MSE.
                # Tie break = mean MAE.
                if (
                    best_row is None
                    or (
                        row["mse_mean"],
                        row["mae_mean"],
                    )
                    <
                    (
                        best_row["mse_mean"],
                        best_row["mae_mean"],
                    )
                ):
                    best_row = dict(row)

                    # Keep only the current best
                    # prediction in RAM.
                    best_raw = (
                        raw_pred.copy()
                    )

                    best_refined = (
                        refined.copy()
                    )

                    best_per_sample = {
                        "mse":
                            mse_v.copy(),
                        "mae":
                            mae_v.copy(),
                        "psnr":
                            psnr_v.copy(),
                    }

    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    ranking = sorted(
        rows,
        key=lambda x: (
            x["mse_mean"],
            x["mae_mean"],
        )
    )

    for rank, row in enumerate(
        ranking,
        start=1,
    ):
        row["rank"] = rank

    print()
    print("=" * 140)
    print(
        "HP-C4 JOINT VAL400 RANKING"
    )
    print("=" * 140)

    print(
        f"{'rank':>4} "
        f"{'step':>6} "
        f"{'sigma':>12} "
        f"{'alpha':>7} "
        f"{'MSE':>10} "
        f"{'MAE':>9} "
        f"{'PSNR':>9} "
        f"{'MSEwin':>9} "
        f"{'MAEwin':>9} "
        f"{'gain':>10} "
        f"{'gain%':>8}"
    )

    for row in ranking[:30]:

        print(
            f"{row['rank']:4d} "
            f"{row['step']:6d} "
            f"{row['sigma']:12.8f} "
            f"{row['alpha']:7.3f} "
            f"{row['mse_mean']:10.4f} "
            f"{row['mae_mean']:9.4f} "
            f"{row['psnr_mean']:9.4f} "
            f"{row['mse_wins']:5d}/400 "
            f"{row['mae_wins']:5d}/400 "
            f"{row['mse_gain_vs_hint']:+10.4f} "
            f"{row['mse_gain_pct']:+7.3f}%"
        )

    print()
    print("=" * 100)
    print("BEST HP-C4")
    print("=" * 100)

    print(
        json.dumps(
            best_row,
            indent=2,
        )
    )

    print()
    print(
        "BASELINE MSE =",
        baseline["mse_mean"],
    )

    # --------------------------------------------------
    # Save best predictions for next-stage
    # adaptive / gated residual experiments.
    # --------------------------------------------------

    np.save(
        out /
        "best_raw_pred_norm.npy",
        best_raw,
    )

    np.save(
        out /
        "best_refined_pred_norm.npy",
        best_refined,
    )

    np.save(
        out /
        "fixed_noise.npy",
        noise,
    )

    # --------------------------------------------------
    # Best per-sample table
    # --------------------------------------------------

    best_rows = []

    for i in range(len(gt)):

        best_rows.append({
            "sample":
                i,

            "hint_mse":
                float(
                    baseline_mse[i]
                ),

            "hp_mse":
                float(
                    best_per_sample[
                        "mse"
                    ][i]
                ),

            "mse_gain":
                float(
                    baseline_mse[i]
                    -
                    best_per_sample[
                        "mse"
                    ][i]
                ),

            "hint_mae":
                float(
                    baseline_mae[i]
                ),

            "hp_mae":
                float(
                    best_per_sample[
                        "mae"
                    ][i]
                ),

            "hint_psnr":
                float(
                    baseline_psnr[i]
                ),

            "hp_psnr":
                float(
                    best_per_sample[
                        "psnr"
                    ][i]
                ),
        })

    # --------------------------------------------------
    # Save summaries
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

                    "test_used":
                        False,

                    "selection_metric":
                        "mean speed-domain MSE",

                    "tie_break":
                        "mean speed-domain MAE",

                    "initialization":
                        "x_t = hint + sigma * epsilon",

                    "refinement":
                        "hint + alpha * "
                        "(prediction - hint)",

                    "seed":
                        args.seed,
                },

                "baseline":
                    baseline,

                "best":
                    best_row,

                "ranking":
                    ranking,
            },
            f,
            indent=2,
        )

    with open(
        out / "summary.csv",
        "w",
        newline="",
    ) as f:

        fields = [
            "rank",
            "step",
            "sigma",
            "alpha",
            "mse_mean",
            "mse_std",
            "mae_mean",
            "psnr_mean",
            "mse_wins",
            "mae_wins",
            "mse_gain_vs_hint",
            "mse_gain_pct",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            ranking
        )

    with open(
        out /
        "best_per_sample.csv",
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                best_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            best_rows
        )

    print()
    print(
        "[PASS] HP-C4 joint "
        "VAL400 selection complete."
    )


if __name__ == "__main__":
    main()
