import argparse
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
        th.load(backbone, map_location="cpu"),
        strict=True,
    )

    control_net.load_state_dict(
        th.load(control, map_location="cpu"),
        strict=True,
    )

    controlled_unet.to(device)
    control_net.to(device)

    controlled_unet.convert_to_fp16()
    control_net.convert_to_fp16()

    controlled_unet.eval()
    control_net.eval()

    return control_net, controlled_unet, diffusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--hint", required=True)
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    gt = np.load(args.gt).astype(np.float32)[:4]
    hint = np.load(args.hint).astype(np.float32)[:4]

    zero = np.zeros_like(hint)
    shuffled = np.roll(hint, 1, axis=0)

    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal(gt.shape).astype(np.float32)

    device = th.device("cuda")

    control, model, diffusion = create_models(
        args.backbone,
        args.control,
        device,
    )

    gt_t = th.from_numpy(gt[:, None]).to(device)
    noise_t = th.from_numpy(noise[:, None]).to(device)

    variants = {
        "correct": hint,
        "zero": zero,
        "shuffled": shuffled,
    }

    for sigma_value in [80.0, 10.0, 1.0, 0.1]:
        sigma = th.full(
            (len(gt),),
            sigma_value,
            dtype=th.float32,
            device=device,
        )

        # diagnostic uses actual noisy GT, matching training x_t construction
        x_t = (
            gt_t +
            noise_t *
            sigma[:, None, None, None]
        )

        outs = {}

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

                outs[name] = (
                    y.float().cpu().numpy()[:, 0]
                )

        print("\n" + "=" * 80)
        print("sigma =", sigma_value)

        for other in ["zero", "shuffled"]:
            d = np.abs(
                outs["correct"] -
                outs[other]
            )

            print(
                f"correct vs {other}: "
                f"mean_abs_norm={d.mean():.8e}, "
                f"mean_abs_mps={d.mean()*SCALE:.6f}, "
                f"max_abs_mps={d.max()*SCALE:.6f}"
            )

        # Correct-hint reconstruction MSE in normalized space.
        for name in variants:
            e = outs[name] - gt
            print(
                f"{name:9s} "
                f"norm_MSE={np.mean(e**2):.8f}"
            )

    print("\n[PASS] multisigma output sensitivity complete")


if __name__ == "__main__":
    main()
