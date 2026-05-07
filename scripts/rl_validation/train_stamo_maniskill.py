"""Train StaMo on ManiSkill3 data for RL validation experiment.

Group A: diffusion-only (semantic_head disabled)
Group B: diffusion + semantic (semantic_head enabled, lambda=0.1)

Usage:
    cd "stamo_pro - 副本 - 副本"
    py rl_validation/scripts/train_stamo_maniskill.py --group A --task PickCube-v1
    py rl_validation/scripts/train_stamo_maniskill.py --group B --task PickCube-v1
    py rl_validation/scripts/train_stamo_maniskill.py --group A --task StackCube-v1
    py rl_validation/scripts/train_stamo_maniskill.py --group B --task StackCube-v1
    py rl_validation/scripts/train_stamo_maniskill.py --group A --task PickCube-v1 --test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[2]
STAMO_ROOT = ROOT / "StaMo"
RL_ROOT = ROOT / "rl_validation"
if str(STAMO_ROOT) not in sys.path:
    sys.path.insert(0, str(STAMO_ROOT))

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from stamo.renderer.model.renderer import RenderNet
from stamo.renderer.utils.data import PairImageData, ImageData, collate_fn, collate_fn_pair
from stamo.renderer.utils.optim import (
    WarmupLinearConstantLR,
    WarmupLinearLR,
    get_criterion,
    get_optimizer,
)

TASK_DATA = {
    "PickCube-v1": "pickcube_v1",
    "StackCube-v1": "stackcube_v1",
}


def vram_gb():
    return torch.cuda.memory_allocated() / 1024**3


def vram_peak_gb():
    return torch.cuda.max_memory_allocated() / 1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=["A", "B"],
                        help="A=diffusion-only, B=diffusion+semantic")
    parser.add_argument("--task", required=True, choices=list(TASK_DATA.keys()))
    parser.add_argument("--test", action="store_true", help="Dry run: 10 steps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_vram_gb", type=float, default=7.0)
    parser.add_argument("--num_iters", type=int, default=None)
    cli = parser.parse_args()

    config_path = RL_ROOT / "configs" / "stamo_maniskill.yaml"
    args = OmegaConf.load(str(config_path))
    args.deepspeed = False

    task_slug = TASK_DATA[cli.task]
    data_dir = RL_ROOT / "data" / task_slug

    if cli.group == "A":
        args.task_name = f"maniskill_{task_slug}_diffonly"
        args.semantic_head.enabled = False
        args.semantic_head.lambda_weight = 0.0
        args.data.train_manifest = str(data_dir / "train.jsonl")
        args.data.eval_manifest = str(data_dir / "test.jsonl")
    else:
        args.task_name = f"maniskill_{task_slug}_semantic"
        args.semantic_head.enabled = True
        args.semantic_head.lambda_weight = 0.1
        args.data.train_manifest = str(data_dir / "train.jsonl")
        args.data.eval_manifest = str(data_dir / "test.jsonl")

    if cli.num_iters is not None:
        args.train.num_iters = cli.num_iters

    if cli.test:
        args.train.num_iters = 10
        args.train.eval_step = 5
        args.train.save_step = 10

    if cli.resume:
        import glob
        ckpt_dir = STAMO_ROOT / args.train.ckpt_save_dir / args.task_name
        steps = sorted([int(Path(d).name) for d in glob.glob(str(ckpt_dir / "*"))
                        if Path(d).name.isdigit()])
        if steps:
            args.resume = True
            args.resume_path = str(ckpt_dir / str(steps[-1]))
            print(f"Resuming from step {steps[-1]}")
        else:
            print("No checkpoint found, starting fresh")

    use_pair = args.semantic_head.enabled
    print(f"Group {cli.group}: {'diffusion+semantic' if use_pair else 'diffusion-only'}")
    print(f"Task: {cli.task}, data: {data_dir}")
    print(f"Config: batch={args.train.local_batch_size}, "
          f"accum={args.train.gradient_accumulate_steps}, "
          f"iters={args.train.num_iters}")

    # Build model
    print("\nBuilding model...")
    model = RenderNet(args)
    model = model.to("cuda")
    model.set_trainable_params()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable/1e6:.1f}M, Frozen: {frozen/1e6:.1f}M")
    print(f"VRAM after model: {vram_gb():.2f} GB")

    optimizer = get_optimizer(
        (p for p in model.parameters() if p.requires_grad),
        opt_type="AdamW", lr=args.train.learning_rate,
        betas=(0.9, 0.98), weight_decay=args.train.decay,
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

    # Data — always use PairImageData (both groups need image pairs for diffusion)
    print("\nLoading data...")
    dataset = PairImageData(
        args.data.train_manifest, flip_p=args.data.flip_p, img_size=args.data.img_size)
    train_loader = DataLoader(
        dataset, batch_size=args.train.local_batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn_pair, drop_last=True)

    print(f"Dataset: {len(dataset)} samples, {len(train_loader)} batches/epoch")
    print(f"VRAM after data: {vram_gb():.2f} GB")

    # Training loop
    print(f"\nTraining from step {global_step + 1}...")
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
            inputs = {
                "images_t": batch["images_t"].to("cuda"),
                "images_tp": batch["images_tp"].to("cuda"),
                "generator": generator,
            }

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
            print(f"\n[WARNING] VRAM {peak:.2f} GB > limit {cli.max_vram_gb} GB")
            save_dir = str(ckpt_dir / f"{step}")
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(save_dir, step)
            return

        if step % 10 == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / 10 * grad_accum
            sps = 10 / elapsed
            msg = (f"  step {step}/{num_iters}  loss={avg_loss:.4f}  "
                   f"VRAM={vram_gb():.2f}/{peak:.2f}GB  "
                   f"{sps:.1f} step/s  lr={scheduler.get_last_lr()[0]:.2e}")
            if use_pair and running_diff > 0:
                avg_d = running_diff / 10 * grad_accum
                avg_s = running_sem / 10 * grad_accum
                msg += f"  diff={avg_d:.4f}  sem={avg_s:.4f}"
                running_diff = 0.0
                running_sem = 0.0
            print(msg)
            running_loss = 0.0
            t0 = time.time()

        if step % args.train.save_step == 0:
            save_dir = str(ckpt_dir / f"{step}")
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(save_dir, step)
            print(f"  [Checkpoint: {save_dir}]")

    save_dir = str(ckpt_dir / f"{num_iters}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(save_dir, num_iters)
    print(f"\nDone! Checkpoint: {save_dir}")
    print(f"Peak VRAM: {vram_peak_gb():.2f} GB")


if __name__ == "__main__":
    main()
