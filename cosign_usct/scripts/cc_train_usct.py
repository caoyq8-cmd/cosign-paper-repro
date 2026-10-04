"""
USCT-specific Conditional Consistency Model training.

Key difference from official CoSIGN cc_train.py:
    hint is loaded from the precomputed OOF InversionNet cache,
    rather than generated online by condition_operator(x_start).

The official ControlNet / ControlledUnet / Karras control loss are reused.
"""

import argparse
import functools
import random

import numpy as np
import torch as th
import torch.distributed as dist

from mpi4py import MPI
from torch.utils.data import Dataset, DataLoader

from cc import dist_util, logger
from cc.resample import (
    create_named_schedule_sampler,
    LossAwareSampler,
)
from cc.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cc_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
)
from cc.train_util import CCTrainLoop, log_loss_dict


class PairedNpyDataset(Dataset):
    def __init__(self, gt_path, hint_path):
        super().__init__()

        self.gt = np.load(gt_path, mmap_mode="r")
        self.hint = np.load(hint_path, mmap_mode="r")

        if self.gt.shape != self.hint.shape:
            raise RuntimeError(
                f"GT/hint shape mismatch: {self.gt.shape} vs {self.hint.shape}"
            )

        if self.gt.ndim != 3:
            raise RuntimeError(
                f"Expected [N,H,W], got {self.gt.shape}"
            )

        if self.gt.shape[1:] != (256, 256):
            raise RuntimeError(
                f"Expected 256x256, got {self.gt.shape}"
            )

        rank = MPI.COMM_WORLD.Get_rank()
        world = MPI.COMM_WORLD.Get_size()

        self.indices = np.arange(len(self.gt))[rank::world]

        if len(self.indices) == 0:
            raise RuntimeError(
                f"No samples on MPI rank {rank}/{world}"
            )

        print(
            f"[PairedNpyDataset] "
            f"global_n={len(self.gt)} "
            f"local_n={len(self.indices)} "
            f"rank={rank}/{world}"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])

        gt = np.array(self.gt[idx], dtype=np.float32, copy=True)
        hint = np.array(self.hint[idx], dtype=np.float32, copy=True)

        # [H,W] -> [1,H,W]
        gt = gt[None, ...]
        hint = hint[None, ...]

        return gt, {"hint": hint}


def load_paired_npy(
    gt_path,
    hint_path,
    batch_size,
    num_workers=1,
):
    dataset = PairedNpyDataset(gt_path, hint_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False,
    )

    while True:
        yield from loader


class USCTCCTrainLoop(CCTrainLoop):
    """
    CCTrainLoop variant using cached OOF hints.
    """

    def run_loop(self):
        while self.step < self.total_training_steps:
            batch, cond = next(self.data)

            self.run_step(batch, cond)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if (
                self.step > 0
                and self.step % self.save_interval == 0
            ):
                self.save()

        if self.step % self.save_interval != 0:
            self.save()

        logger.dumpkvs()

    def save(self):
        """Save either the official full state or a lightweight raw checkpoint."""
        if getattr(self, "save_mode", "full") == "full":
            return super().save()

        if getattr(self, "save_mode", "full") != "raw":
            raise ValueError(
                f"Unknown save_mode={self.save_mode}"
            )

        import os
        import torch as th
        from cc.train_util import get_blob_logdir

        if dist.get_rank() == 0:
            state_dict = (
                self.mp_trainer.master_params_to_state_dict(
                    self.mp_trainer.master_params
                )
            )

            path = os.path.join(
                get_blob_logdir(),
                f"model{self.step:06d}.pt",
            )

            logger.log(
                f"saving RAW-ONLY checkpoint to {path}"
            )

            th.save(state_dict, path)

        dist.barrier()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()

        # controlled_unet must keep requires_grad=True for CoSIGN's
        # custom checkpointing, but it is not part of the optimizer.
        for p in self.controlled_unet.parameters():
            p.grad = None

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i:i + self.microbatch].to(
                dist_util.dev()
            )

            micro_cond = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            if "hint" not in micro_cond:
                raise RuntimeError("Paired loader did not provide hint")

            # This is the key USCT modification.
            hint = micro_cond.pop("hint")

            if micro.shape != hint.shape:
                raise RuntimeError(
                    f"x_start/hint mismatch: "
                    f"{micro.shape} vs {hint.shape}"
                )

            last_batch = (
                i + self.microbatch
            ) >= batch.shape[0]

            # Kept for compatibility with official training loop.
            t, weights = self.schedule_sampler.sample(
                micro.shape[0],
                dist_util.dev(),
            )

            num_scales = self.diffusion.num_timesteps

            compute_losses = functools.partial(
                self.diffusion.control_losses,
                self.controlled_unet,
                micro,
                hint,
                num_scales,
                model_kwargs=micro_cond,
                control_model=self.ddp_model,
                target_control_model=None,
                dump_imgs=False,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(
                self.schedule_sampler,
                LossAwareSampler,
            ):
                self.schedule_sampler.update_with_local_losses(
                    t,
                    losses["loss"].detach(),
                )

            loss = (
                losses["loss"] * weights
            ).mean()

            losses_for_log = {
                k: v
                for k, v in losses.items()
                if k != "recon"
            }

            log_loss_dict(
                self.diffusion,
                t,
                {
                    k: v * weights
                    for k, v in losses_for_log.items()
                },
            )

            if not th.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite C4 loss: {loss.item()}"
                )

            self.mp_trainer.backward(loss)

            # The gradient through controlled_unet is needed to propagate
            # back into ControlNet, but controlled_unet itself is frozen
            # in the optimizer sense, so discard its parameter gradients.
            for p in self.controlled_unet.parameters():
                p.grad = None


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args=args)

    rank = dist.get_rank()

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    logger.log("===== C4 USCT Conditional CM =====")
    logger.log(f"gt_npy   = {args.gt_npy}")
    logger.log(f"hint_npy = {args.hint_npy}")
    logger.log(f"unet     = {args.unet_path}")
    logger.log(f"steps    = {args.total_training_steps}")
    logger.log(f"batch    = {args.batch_size}")
    logger.log(f"loss     = {args.loss_type}/{args.loss_norm}")

    logger.log("creating ControlNet / ControlledUnet / diffusion...")

    kwargs = args_to_dict(
        args,
        model_and_diffusion_defaults().keys(),
    )

    kwargs["distillation"] = True
    kwargs["control"] = True

    control_net, controlled_unet, diffusion = (
        create_model_and_diffusion(**kwargs)
    )

    if not args.unet_path:
        raise RuntimeError(
            "C4 requires the pretrained C3 CM checkpoint"
        )

    logger.log(
        f"loading C3 backbone from {args.unet_path}"
    )

    state = dist_util.load_state_dict(
        args.unet_path,
        map_location="cpu",
    )

    incompat = control_net.load_state_dict(
        state,
        strict=False,
    )

    logger.log(
        f"ControlNet init: "
        f"missing={len(incompat.missing_keys)}, "
        f"unexpected={len(incompat.unexpected_keys)}"
    )

    # Official CoSIGN loads the CM weights strictly
    # into the controlled U-Net.
    controlled_unet.load_state_dict(
        state,
        strict=True,
    )

    control_net.to(dist_util.dev())
    controlled_unet.to(dist_util.dev())

    # IMPORTANT:
    # CoSIGN's custom gradient checkpointing passes module parameters
    # into CheckpointFunction. They must therefore keep requires_grad=True.
    #
    # This does NOT mean controlled_unet is optimized:
    # CCTrainLoop's optimizer is constructed only from model=control_net.
    for p in controlled_unet.parameters():
        p.requires_grad_(True)

    controlled_unet.train()
    control_net.train()

    if args.use_fp16:
        control_net.convert_to_fp16()
        controlled_unet.convert_to_fp16()

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler,
        diffusion,
    )

    if args.batch_size == -1:
        batch_size = (
            args.global_batch_size
            // dist.get_world_size()
        )
    else:
        batch_size = args.batch_size

    if batch_size < 1:
        raise RuntimeError("batch_size < 1")

    logger.log("creating paired GT/OOF-hint loader...")

    data = load_paired_npy(
        args.gt_npy,
        args.hint_npy,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    # Inspect one batch before optimization.
    b0, c0 = next(data)

    logger.log(
        f"preflight GT   shape={tuple(b0.shape)} "
        f"range=({float(b0.min()):.6f},"
        f"{float(b0.max()):.6f})"
    )

    logger.log(
        f"preflight hint shape={tuple(c0['hint'].shape)} "
        f"range=({float(c0['hint'].min()):.6f},"
        f"{float(c0['hint'].max()):.6f})"
    )

    logger.log("training...")

    loop = USCTCCTrainLoop(
        model=control_net,
        controlled_unet=controlled_unet,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        condition_operator=None,
        noiser=None,
    )

    loop.save_mode = args.save_mode
    logger.log(f"save_mode = {args.save_mode}")

    loop.run_loop()

    logger.log("[PASS] C4 training completed.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def create_argparser():
    defaults = dict(
        gt_npy="",
        hint_npy="",
        num_workers=1,
        seed=42,
        save_mode="full",

        schedule_sampler="uniform",
        lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,

        global_batch_size=1,
        batch_size=1,
        microbatch=1,

        ema_rate="0.9999",
        log_interval=1,
        save_interval=25,

        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,

        wandb_api_key="",
        wandb_user="",
        name="",
    )

    defaults.update(
        model_and_diffusion_defaults()
    )

    defaults.update(
        cc_train_defaults()
    )

    parser = argparse.ArgumentParser()

    add_dict_to_argparser(
        parser,
        defaults,
    )

    return parser


if __name__ == "__main__":
    main()

