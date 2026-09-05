import argparse
import csv
from pathlib import Path

import numpy as np
import torch as th

from cc.script_util import create_model_and_diffusion


SCALE = 102.5


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
        "--tag",
        default="c4",
    )

    ap.add_argument(
        "--num_samples",
        type=int,
        default=8,
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

    gt_all = np.load(
        args.gt
    ).astype(np.float32)

    hint_all = np.load(
        args.hint
    ).astype(np.float32)

    if gt_all.shape != hint_all.shape:
        raise RuntimeError(
            f"GT/hint mismatch: "
            f"{gt_all.shape} vs "
            f"{hint_all.shape}"
        )

    if len(gt_all) < args.num_samples:
        raise RuntimeError(
            f"Requested {args.num_samples} samples, "
            f"but dataset has {len(gt_all)}"
        )

    gt = gt_all[
        :args.num_samples
    ]

    hint = hint_all[
        :args.num_samples
    ]

    print(
        "audit samples =",
        len(gt),
    )

    print(
        "GT shape      =",
        gt.shape,
    )

    zero = np.zeros_like(
        hint
    )

    shuffled = np.roll(
        hint,
        shift=1,
        axis=0,
    )

    rng = np.random.default_rng(
        args.seed
    )

    noise = rng.standard_normal(
        gt.shape
    ).astype(np.float32)

    device = th.device(
        "cuda"
        if th.cuda.is_available()
        else "cpu"
    )

    (
        control,
        model,
        diffusion,
    ) = create_models(
        args.backbone,
        args.control,
        device,
    )

    gt_t = th.from_numpy(
        gt[:, None]
    ).to(device)

    noise_t = th.from_numpy(
        noise[:, None]
    ).to(device)

    variants = {
        "correct": hint,
        "zero": zero,
        "shuffled": shuffled,
    }

    rows = []

    for sigma_value in [
        80.0,
        10.0,
        1.0,
        0.1,
    ]:

        sigma = th.full(
            (len(gt),),
            sigma_value,
            dtype=th.float32,
            device=device,
        )

        # Match training-style noisy input.
        x_t = (
            gt_t
            +
            noise_t
            *
            sigma[:, None, None, None]
        )

        outputs = {}

        with th.no_grad():

            for name, h_np in variants.items():

                h = th.from_numpy(
                    h_np[:, None]
                ).to(device)

                _, y = diffusion.recon(
                    model,
                    control,
                    x_t,
                    h,
                    sigma,
                )

                outputs[name] = (
                    y.float()
                    .cpu()
                    .numpy()[:, 0]
                )

        print()
        print("=" * 110)
        print(
            f"{args.tag} | sigma={sigma_value:g}"
        )
        print("=" * 110)

        wins_zero = 0
        wins_shuffle = 0

        aggregate = {
            "correct": [],
            "zero": [],
            "shuffled": [],
        }

        for i in range(len(gt)):

            mse = {}

            for name in variants:

                e = (
                    outputs[name][i]
                    -
                    gt[i]
                )

                mse[name] = float(
                    np.mean(e ** 2)
                )

                aggregate[
                    name
                ].append(
                    mse[name]
                )

            d_zero = np.abs(
                outputs["correct"][i]
                -
                outputs["zero"][i]
            )

            d_shuffle = np.abs(
                outputs["correct"][i]
                -
                outputs["shuffled"][i]
            )

            cz_mean_mps = float(
                d_zero.mean()
                *
                SCALE
            )

            cs_mean_mps = float(
                d_shuffle.mean()
                *
                SCALE
            )

            gain_zero = (
                mse["zero"]
                -
                mse["correct"]
            )

            gain_shuffle = (
                mse["shuffled"]
                -
                mse["correct"]
            )

            win_zero = (
                gain_zero > 0
            )

            win_shuffle = (
                gain_shuffle > 0
            )

            wins_zero += int(
                win_zero
            )

            wins_shuffle += int(
                win_shuffle
            )

            rows.append({
                "tag":
                    args.tag,

                "sigma":
                    sigma_value,

                "sample":
                    i,

                "correct_mse":
                    mse["correct"],

                "zero_mse":
                    mse["zero"],

                "shuffled_mse":
                    mse["shuffled"],

                "gain_vs_zero":
                    gain_zero,

                "gain_vs_shuffle":
                    gain_shuffle,

                "correct_zero_mean_mps":
                    cz_mean_mps,

                "correct_shuffle_mean_mps":
                    cs_mean_mps,

                "win_vs_zero":
                    int(win_zero),

                "win_vs_shuffle":
                    int(win_shuffle),
            })

            print(
                f"sample={i:02d} | "
                f"C={mse['correct']:.8f} "
                f"Z={mse['zero']:.8f} "
                f"S={mse['shuffled']:.8f} | "
                f"G-Z={gain_zero:+.8e} "
                f"G-S={gain_shuffle:+.8e} | "
                f"C-Z={cz_mean_mps:.6f} m/s "
                f"C-S={cs_mean_mps:.6f} m/s | "
                f"winZ={win_zero} "
                f"winS={win_shuffle}"
            )

        print()
        print("-" * 110)

        for name in [
            "correct",
            "zero",
            "shuffled",
        ]:

            a = np.asarray(
                aggregate[name],
                dtype=np.float64,
            )

            print(
                f"{name:9s} "
                f"MSE="
                f"{a.mean():.8f}"
                f" ± "
                f"{a.std():.8f}"
            )

        print(
            f"correct wins vs zero     = "
            f"{wins_zero}/{len(gt)}"
        )

        print(
            f"correct wins vs shuffled = "
            f"{wins_shuffle}/{len(gt)}"
        )

        gz = (
            np.mean(
                aggregate["zero"]
            )
            -
            np.mean(
                aggregate["correct"]
            )
        )

        gs = (
            np.mean(
                aggregate["shuffled"]
            )
            -
            np.mean(
                aggregate["correct"]
            )
        )

        print(
            f"aggregate gain vs zero     = "
            f"{gz:+.8e}"
        )

        print(
            f"aggregate gain vs shuffled = "
            f"{gs:+.8e}"
        )

    csv_path = (
        out
        /
        f"{args.tag}_per_sample8.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        "[PASS] per-sample C4 audit complete"
    )

    print(
        "CSV =",
        csv_path,
    )


if __name__ == "__main__":
    main()
