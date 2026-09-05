import argparse
from pathlib import Path

import numpy as np
import torch as th

from cc.script_util import create_model_and_diffusion
from cc.karras_diffusion import append_dims


def group_name(k):
    for name in [
        "input_hint_block",
        "zero_convs",
        "middle_block_out",
        "input_blocks",
        "middle_block",
        "time_embed",
    ]:
        if k.startswith(name):
            return name
    return "other"


def tensor_rms(x):
    x = x.float()
    return float(th.sqrt(th.mean(x * x)).item())


def aggregate_rms(xs):
    ss = 0.0
    n = 0
    for x in xs:
        y = x.float()
        ss += float((y * y).sum().item())
        n += y.numel()
    return (ss / max(n, 1)) ** 0.5


def aggregate_diff_rms(a, b):
    ss = 0.0
    n = 0
    for x, y in zip(a, b):
        d = x.float() - y.float()
        ss += float((d * d).sum().item())
        n += d.numel()
    return (ss / max(n, 1)) ** 0.5


def create_control(device):
    control, _, diffusion = create_model_and_diffusion(
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
        use_fp16=False,
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

    control.to(device)
    control.eval()

    return control, diffusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control50", required=True)
    ap.add_argument("--control500", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--hint", required=True)
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    # ============================================================
    # A. PARAMETER DELTA: raw50 -> raw500
    # ============================================================

    s50 = th.load(args.control50, map_location="cpu")
    s500 = th.load(args.control500, map_location="cpu")

    if set(s50) != set(s500):
        print("[WARN] state_dict key sets differ")

    groups = {}

    for k in sorted(set(s50) & set(s500)):
        if not th.is_tensor(s50[k]):
            continue

        g = group_name(k)

        groups.setdefault(
            g,
            {
                "delta_ss": 0.0,
                "current_ss": 0.0,
                "n": 0,
                "keys": 0,
            },
        )

        a = s50[k].float()
        b = s500[k].float()
        d = b - a

        groups[g]["delta_ss"] += float((d * d).sum())
        groups[g]["current_ss"] += float((b * b).sum())
        groups[g]["n"] += b.numel()
        groups[g]["keys"] += 1

    print("\n" + "=" * 90)
    print("PARAMETER DELTA: RAW50 -> RAW500")
    print("=" * 90)

    for g, z in groups.items():
        delta_rms = (
            z["delta_ss"] / max(z["n"], 1)
        ) ** 0.5

        current_rms = (
            z["current_ss"] / max(z["n"], 1)
        ) ** 0.5

        ratio = delta_rms / (
            current_rms + 1e-20
        )

        print(
            f"{g:20s} "
            f"keys={z['keys']:3d} "
            f"delta_rms={delta_rms:.6e} "
            f"current_rms={current_rms:.6e} "
            f"delta/current={ratio:.6e}"
        )

    # ============================================================
    # B. CONTROL FEATURE SENSITIVITY
    # ============================================================

    gt = np.load(args.gt).astype(np.float32)
    hint = np.load(args.hint).astype(np.float32)

    if gt.shape != hint.shape:
        raise RuntimeError(
            f"shape mismatch: {gt.shape} vs {hint.shape}"
        )

    # audit at most first four
    gt = gt[:4]
    hint = hint[:4]

    zero = np.zeros_like(hint)
    shuffled = np.roll(hint, 1, axis=0)

    device = th.device("cuda")

    control, diffusion = create_control(device)

    control.load_state_dict(
        s500,
        strict=True,
    )

    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal(
        gt.shape
    ).astype(np.float32)

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

    print("\n" + "=" * 90)
    print("CONTROL FEATURE SENSITIVITY")
    print("=" * 90)

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

        # exactly matches training form x_t = x_start + noise * sigma
        x_t = (
            gt_t
            + noise_t
            * append_dims(
                sigma,
                gt_t.ndim,
            )
        )

        _, _, c_in = [
            append_dims(
                x,
                x_t.ndim,
            )
            for x in
            diffusion.get_scalings_for_boundary_condition(
                sigma
            )
        ]

        rescaled_t = (
            1000
            * 0.25
            * th.log(
                sigma + 1e-44
            )
        )

        outputs = {}

        with th.no_grad():
            for name, h_np in variants.items():
                h = th.from_numpy(
                    h_np[:, None]
                ).to(device)

                outputs[name] = control(
                    c_in * x_t,
                    c_in * h,
                    rescaled_t,
                )

        rms_correct = aggregate_rms(
            outputs["correct"]
        )

        dz = aggregate_diff_rms(
            outputs["correct"],
            outputs["zero"],
        )

        ds = aggregate_diff_rms(
            outputs["correct"],
            outputs["shuffled"],
        )

        print()
        print(f"sigma={sigma_value:g}")
        print(
            " control RMS                =",
            f"{rms_correct:.6e}",
        )
        print(
            " correct-zero diff RMS      =",
            f"{dz:.6e}",
        )
        print(
            " correct-shuffled diff RMS  =",
            f"{ds:.6e}",
        )
        print(
            " correct-zero relative      =",
            f"{dz/(rms_correct+1e-20):.6e}",
        )
        print(
            " correct-shuffled relative  =",
            f"{ds/(rms_correct+1e-20):.6e}",
        )

        if sigma_value == 80.0:
            print("\n  per-layer @ sigma80")

            for i, (
                c,
                z,
                s,
            ) in enumerate(
                zip(
                    outputs["correct"],
                    outputs["zero"],
                    outputs["shuffled"],
                )
            ):
                rc = tensor_rms(c)
                rz = tensor_rms(c - z)
                rs = tensor_rms(c - s)

                print(
                    f"  layer={i:02d} "
                    f"RMS={rc:.6e} "
                    f"diff_zero={rz:.6e} "
                    f"diff_shuffle={rs:.6e}"
                )

    print("\n[PASS] control-path audit complete")


if __name__ == "__main__":
    main()

