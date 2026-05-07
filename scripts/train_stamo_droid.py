"""Safe StaMo training script with semantic head + VRAM monitoring.

Supports two modes:
  - Pair mode (default with semantic_head.enabled): loads image pairs + semantic labels
  - Legacy mode: loads single images (original behavior)

Usage:
    cd StaMo && py ../scripts/train_stamo_droid.py --test        # dry run: 10 steps
    cd StaMo && py ../scripts/train_stamo_droid.py               # full training
    cd StaMo && py ../scripts/train_stamo_droid.py --resume      # resume from last ckpt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[1]
STAMO_ROOT = ROOT / "StaMo"
if str(STAMO_ROOT) not in sys.path:
    sys.path.insert(0, str(STAMO_ROOT))

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from stamo.renderer.model.renderer import RenderNet
from stamo.renderer.utils.data import ImageData, PairImageData, collate_fn, collate_fn_pair
from stamo.renderer.utils.optim import (
    WarmupLinearConstantLR,
    WarmupLinearLR,
    get_criterion,
    get_optimizer,
)


def vram_gb():
    return torch.cuda.memory_allocated() / 1024**3


def vram_peak_gb():
    return torch.cuda.max_memory_allocated() / 1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=STAMO_ROOT / "configs" / "droid.yaml")
    parser.add_argument("--test", action="store_true", help="Dry run: 10 steps only")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--max_vram_gb", type=float, default=7.0, help="VRAM safety limit")
    cli = parser.parse_args()

    args = OmegaConf.load(str(cli.config))
    args.deepspeed = False

    if cli.test:
        args.train.num_iters = 10
        args.train.eval_step = 5
        args.train.save_step = 10

    if cli.resume:
        import glob
        ckpt_dir = STAMO_ROOT / args.train.ckpt_save_dir / args.task_name
        steps = sorted([int(Path(d).name) for d in glob.glob(str(ckpt_dir / "*")) if Path(d).name.isdigit()])
        if steps:
            args.resume = True
            args.resume_path = str(ckpt_dir / str(steps[-1]))
            print(f"Resuming from step {steps[-1]}")
        else:
            print("No checkpoint found, starting fresh")

    use_pair = getattr(args, "semantic_head", None) is not None and getattr(args.semantic_head, "enabled", False)
    print(f"Mode: {'pair (semantic+diffusion)' if use_pair else 'single-image (diffusion only)'}")
    print(f"Config: batch_size={args.train.local_batch_size}, "
          f"grad_accum={args.train.gradient_accumulate_steps}, "
          f"effective_batch={args.train.local_batch_size * args.train.gradient_accumulate_steps}")
    print(f"Iterations: {args.train.num_iters}, img_size={args.data.img_size}")

    # Build model
    print("\nBuilding model...")
    model = RenderNet(args)
    model = model.to("cuda")
    model.set_trainable_params()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable/1e6:.1f}M, Frozen: {frozen/1e6:.1f}M")
    print(f"VRAM after model load: {vram_gb():.2f} GB")

    optimizer = get_optimizer(
        (p for p in model.parameters() if p.requires_grad),
        opt_type="AdamW",
        lr=args.train.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=args.train.decay,
    )

    criterion = get_criterion(loss_type="diffusion", reduction="mean")

    num_iters = args.train.num_iters
    grad_accum = args.train.gradient_accumulate_steps

    if args.train.constant_lr:
        scheduler = WarmupLinearConstantLR(optimizer, max_iter=(num_iters // grad_accum) + 1)
    else:
        scheduler = WarmupLinearLR(optimizer, max_iter=num_iters)

    global_step = 0
    if getattr(args, "resume", False) and args.resume_path:
        global_step = model.load_checkpoint(args.resume_path)
        print(f"Loaded checkpoint at step {global_step}")

    # Data
    print("\nLoading data...")
    if use_pair:
        manifest_path = args.data.train_manifest
        dataset = PairImageData(manifest_path, flip_p=args.data.flip_p, img_size=args.data.img_size)
        train_loader = DataLoader(
            dataset,
            batch_size=args.train.local_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_pair,
            drop_last=True,
        )
    else:
        import json as _json
        json_path = args.data.train_json_path
        with open(json_path, "r") as f:
            config = _json.load(f)
        dataset_path = os.path.join(os.path.dirname(json_path), config["datasets"][0])
        dataset = ImageData(dataset_path, flip_p=args.data.flip_p, img_size=args.data.img_size)
        train_loader = DataLoader(
            dataset,
            batch_size=args.train.local_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=True,
        )

    print(f"Dataset: {len(dataset)} samples, {len(train_loader)} batches/epoch")
    print(f"VRAM after data load: {vram_gb():.2f} GB")

    # Training loop
    print(f"\nStarting training from step {global_step + 1}...")
    model.train()

    def infinite_loader():
        while True:
            yield from train_loader

    train_iter = iter(infinite_loader())

    torch.cuda.reset_peak_memory_stats()
    running_loss = 0.0
    running_diff = 0.0
    running_sem = 0.0
    t0 = time.time()

    ckpt_dir = Path(STAMO_ROOT / args.train.ckpt_save_dir / args.task_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    optimizer.zero_grad()

    for step in range(global_step + 1, num_iters + 1):
        batch = next(train_iter)

        if use_pair:
            inputs = {
                "images_t": batch["images_t"].to("cuda"),
                "images_tp": batch["images_tp"].to("cuda"),
                "labels": {k: v.to("cuda") for k, v in batch["labels"].items()},
                "generator": generator,
            }
        else:
            inputs = {"images": batch["images"].to("cuda"), "generator": generator}

        outputs = model.train_step(inputs, {}, criterion=criterion)
        loss = outputs["loss"] / grad_accum
        loss.backward()

        running_loss += loss.item()
        if "loss_diffusion" in outputs:
            running_diff += outputs["loss_diffusion"].item() / grad_accum
            running_sem += outputs["loss_semantic"].item() / grad_accum

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        peak = vram_peak_gb()
        if peak > cli.max_vram_gb:
            print(f"\n[WARNING] VRAM peak {peak:.2f} GB > limit {cli.max_vram_gb} GB")
            print("Saving emergency checkpoint and stopping...")
            save_dir = str(ckpt_dir / f"{step}")
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(save_dir, step)
            return

        if step % 10 == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / 10 * grad_accum
            steps_per_sec = 10 / elapsed
            log_msg = (f"  step {step}/{num_iters}  loss={avg_loss:.4f}  "
                       f"VRAM={vram_gb():.2f}/{peak:.2f}GB  "
                       f"speed={steps_per_sec:.1f} step/s  "
                       f"lr={scheduler.get_last_lr()[0]:.2e}")
            if use_pair:
                avg_diff = running_diff / 10 * grad_accum
                avg_sem = running_sem / 10 * grad_accum
                log_msg += f"  diff={avg_diff:.4f}  sem={avg_sem:.4f}"
                running_diff = 0.0
                running_sem = 0.0
            print(log_msg)
            running_loss = 0.0
            t0 = time.time()

        if step % args.train.save_step == 0:
            save_dir = str(ckpt_dir / f"{step}")
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(save_dir, step)
            print(f"  [Checkpoint saved: {save_dir}]")

    save_dir = str(ckpt_dir / f"{num_iters}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(save_dir, num_iters)
    print(f"\nTraining complete! Final checkpoint: {save_dir}")
    print(f"Peak VRAM: {vram_peak_gb():.2f} GB")


if __name__ == "__main__":
    main()