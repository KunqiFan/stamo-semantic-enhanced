"""Train lambda=1.0 from scratch for fair ablation comparison.

The original lambda=1.0 was trained by resuming from a pretrained diffusion checkpoint,
giving it an unfair advantage. This script trains it from scratch like the other lambdas.

Usage:
    cd StaMo && py -u ../scripts/train_lam1.0_scratch.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STAMO_ROOT = ROOT / "StaMo"
if str(STAMO_ROOT) not in sys.path:
    sys.path.insert(0, str(STAMO_ROOT))

SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from stamo.renderer.model.renderer import RenderNet
from stamo.renderer.utils.data import PairImageData, collate_fn_pair
from stamo.renderer.utils.optim import (
    WarmupLinearConstantLR, WarmupLinearLR, get_criterion, get_optimizer,
)
import torchvision.transforms as T
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from stamo_bridge.semantics.interface import SEMANTIC_FIELDS

NUM_ITERS = 5000
GRAD_ACCUM = 4
LAM = 1.0
TASK_NAME = "droid_sem_lam1.0"


def train():
    config_path = STAMO_ROOT / "configs" / "droid.yaml"
    print(f"\n{'='*60}")
    print(f"Training lambda={LAM} FROM SCRATCH, task={TASK_NAME}")
    print(f"{'='*60}")

    args = OmegaConf.load(str(config_path))
    args.deepspeed = False
    args.semantic_head.enabled = True
    args.semantic_head.lambda_weight = LAM
    args.task_name = TASK_NAME
    args.train.num_iters = NUM_ITERS
    args.train.save_step = NUM_ITERS

    model = RenderNet(args)
    model = model.to("cuda")
    model.set_trainable_params()

    optimizer = get_optimizer(
        (p for p in model.parameters() if p.requires_grad),
        opt_type="AdamW", lr=args.train.learning_rate,
        betas=(0.9, 0.98), weight_decay=args.train.decay,
    )
    criterion = get_criterion(loss_type="diffusion", reduction="mean")

    if args.train.constant_lr:
        scheduler = WarmupLinearConstantLR(optimizer, max_iter=(NUM_ITERS // GRAD_ACCUM) + 1)
    else:
        scheduler = WarmupLinearLR(optimizer, max_iter=NUM_ITERS)

    dataset = PairImageData(args.data.train_manifest, flip_p=args.data.flip_p, img_size=args.data.img_size)
    train_loader = DataLoader(
        dataset, batch_size=args.train.local_batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn_pair, drop_last=True,
    )

    def infinite_loader():
        while True:
            yield from train_loader

    train_iter = iter(infinite_loader())
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    model.train()
    optimizer.zero_grad()

    running_loss = running_diff = running_sem = 0.0
    t0 = time.time()

    for step in range(1, NUM_ITERS + 1):
        batch = next(train_iter)
        inputs = {
            "images_t": batch["images_t"].to("cuda"),
            "images_tp": batch["images_tp"].to("cuda"),
            "labels": {k: v.to("cuda") for k, v in batch["labels"].items()},
            "generator": generator,
        }

        outputs = model.train_step(inputs, {}, criterion=criterion)
        loss = outputs["loss"] / GRAD_ACCUM
        loss.backward()

        running_loss += loss.item()
        running_diff += outputs["loss_diffusion"].item() / GRAD_ACCUM
        running_sem += outputs["loss_semantic"].item() / GRAD_ACCUM

        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 500 == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / 50 * GRAD_ACCUM
            avg_diff = running_diff / 50 * GRAD_ACCUM
            avg_sem = running_sem / 50 * GRAD_ACCUM
            speed = 500 / elapsed
            print(f"  [{TASK_NAME}] step {step}/{NUM_ITERS}  loss={avg_loss:.4f}  "
                  f"diff={avg_diff:.4f}  sem={avg_sem:.4f}  speed={speed:.1f} step/s")
            running_loss = running_diff = running_sem = 0.0
            t0 = time.time()

    ckpt_dir = STAMO_ROOT / args.train.ckpt_save_dir / TASK_NAME / str(NUM_ITERS)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(ckpt_dir), NUM_ITERS)
    print(f"  Saved checkpoint: {ckpt_dir}")

    del model, optimizer, scheduler, train_iter, dataset, train_loader
    gc.collect()
    torch.cuda.empty_cache()


def extract_features():
    config_path = STAMO_ROOT / "configs" / "droid.yaml"
    ckpt_dir = str(STAMO_ROOT / "ckpts" / TASK_NAME / str(NUM_ITERS))
    out_dir = ROOT / "data" / "interim" / "droid_process_chain" / "latents_sem_lam1.0"

    args = OmegaConf.load(str(config_path))
    args.deepspeed = False
    args.semantic_head.enabled = True

    model = RenderNet(args)
    model.load_checkpoint(ckpt_dir)
    model = model.to("cuda")
    model.eval()

    to_tensor = T.Compose([
        T.Resize((args.data.img_size, args.data.img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])

    out_dir.mkdir(parents=True, exist_ok=True)

    for manifest_path in [
        ROOT / "data" / "processed" / "droid_process_chain" / "train.jsonl",
        ROOT / "data" / "processed" / "droid_process_chain" / "test.jsonl",
    ]:
        rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  Extracting {len(rows)} features from {manifest_path.name}...")

        with torch.no_grad():
            for row in rows:
                img_t = to_tensor(Image.open(row["image_t"]).convert("RGB")).unsqueeze(0).to("cuda")
                img_tp = to_tensor(Image.open(row["image_tp"]).convert("RGB")).unsqueeze(0).to("cuda")

                inp_t = model.projector_feature_extractor(img_t)
                inp_tp = model.projector_feature_extractor(img_tp)

                z_t, pooled_t = model.encode(inp_t)
                z_tp, pooled_tp = model.encode(inp_tp)

                np.savez_compressed(
                    out_dir / f"{row['sample_id']}.npz",
                    delta_z=(z_tp - z_t).squeeze(0).cpu().numpy().astype(np.float32),
                    delta_pooled=(pooled_tp - pooled_t).squeeze(0).cpu().numpy().astype(np.float32),
                    z_t=z_t.squeeze(0).cpu().numpy().astype(np.float32),
                    z_tp=z_tp.squeeze(0).cpu().numpy().astype(np.float32),
                    pooled_t=pooled_t.squeeze(0).cpu().numpy().astype(np.float32),
                    pooled_tp=pooled_tp.squeeze(0).cpu().numpy().astype(np.float32),
                )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Features saved to {out_dir}")


def run_eval():
    latent_dir = ROOT / "data" / "interim" / "droid_process_chain" / "latents_sem_lam1.0"
    train_manifest = ROOT / "data" / "processed" / "droid_process_chain" / "train.jsonl"
    eval_manifest = ROOT / "data" / "processed" / "droid_process_chain" / "test.jsonl"

    train_rows = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in eval_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Visual-only features (2560D)
    def load_visual(row):
        data = np.load(latent_dir / f"{row['sample_id']}.npz")
        dz = data["delta_z"].reshape(-1).astype(np.float32)
        dp = data["delta_pooled"].reshape(-1).astype(np.float32)
        return np.concatenate([dz, dp])

    # Full features (2629D)
    def load_full(row):
        data = np.load(latent_dir / f"{row['sample_id']}.npz")
        dz = data["delta_z"].reshape(-1).astype(np.float32)
        dp = data["delta_pooled"].reshape(-1).astype(np.float32)
        dz_2d = data["delta_z"].astype(np.float32)
        stats = []
        for r in dz_2d:
            stats.extend([r.mean(), r.std(), np.abs(r).max(),
                          np.percentile(r, 25), np.percentile(r, 75),
                          float(np.linalg.norm(r)), float((r > 0).sum()) / len(r)])
        stats_arr = np.array(stats, dtype=np.float32)
        topk = 32
        abs_dz = np.abs(dz)
        topk_idx = np.argsort(abs_dz)[-topk:]
        topk_feats = dz[topk_idx]
        action = np.array(row.get("action", [0]*7), dtype=np.float32)
        ee_delta = np.array(row.get("ee_delta", [0]*6), dtype=np.float32)
        gripper_cmd = action[6] if len(action) > 6 else 0.0
        xyz_norm = float(np.linalg.norm(ee_delta[:3]))
        z_comp = ee_delta[2] if len(ee_delta) > 2 else 0.0
        rot_norm = float(np.linalg.norm(ee_delta[3:6])) if len(ee_delta) > 3 else 0.0
        action_xyz_norm = float(np.linalg.norm(action[:3]))
        gripper_is_closing = float(gripper_cmd < 0.3)
        gripper_is_opening = float(gripper_cmd > 0.7)
        low_movement = float(xyz_norm < 0.01)
        grasp_signal = gripper_is_closing * low_movement
        place_signal = gripper_is_opening * low_movement
        proprio = np.array([*action, *ee_delta, gripper_cmd, xyz_norm, z_comp,
                           rot_norm, action_xyz_norm, gripper_is_closing,
                           gripper_is_opening, low_movement, grasp_signal, place_signal],
                          dtype=np.float32)
        return np.concatenate([dz, dp, stats_arr, topk_feats, proprio])

    for feat_name, loader_fn in [("visual-only (2560D)", load_visual), ("full (2629D)", load_full)]:
        print(f"\n  === lambda=1.0 from scratch — {feat_name} ===")
        x_train = np.stack([loader_fn(r) for r in train_rows])
        x_eval = np.stack([loader_fn(r) for r in eval_rows])
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_eval_s = scaler.transform(x_eval)

        for clf_name, clf in [
            ("HistGBT", HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.03, max_depth=6,
                min_samples_leaf=5, l2_regularization=0.1,
                class_weight="balanced", random_state=42)),
            ("LogReg", LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced",
                random_state=42, solver="lbfgs")),
        ]:
            print(f"  {clf_name}:")
            accs = []
            for target in SEMANTIC_FIELDS:
                y_train = np.asarray([r["labels"][target] for r in train_rows])
                y_eval = np.asarray([r["labels"][target] for r in eval_rows])
                clf.fit(x_train_s, y_train)
                acc = (clf.predict(x_eval_s) == y_eval).mean()
                accs.append(acc)
                print(f"    {target}: {acc:.4f}")
            print(f"    mean: {np.mean(accs):.4f}")


if __name__ == "__main__":
    print("Step 1: Training...")
    train()
    print("\nStep 2: Extracting features...")
    extract_features()
    print("\nStep 3: Evaluation...")
    run_eval()
    print("\nDone!")
