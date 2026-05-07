"""Ablation: SemanticHead insertion position.

Compares 3 positions where SemanticHead reads its delta input:
  A) "pooled"  — after DiTConditionHead (delta_pooled, 512D) [current design]
  B) "proj"    — after Projector (mean-pooled 2 tokens, 1024D)
  C) "dino"    — after DINOv2 VisionBackbone (mean-pooled patches, 768D, frozen)

For each position: train 5000 steps → extract latents → evaluate (visual-only 2560D).

Usage:
    cd StaMo && py -u ../scripts/ablation_head_position.py
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

POSITIONS = {
    "pooled": {"position": "pooled", "task_name": "droid_sem_pos_pooled", "latent_name": "latents_pos_pooled"},
    "proj":   {"position": "proj",   "task_name": "droid_sem_pos_proj",   "latent_name": "latents_pos_proj"},
    "dino":   {"position": "dino",   "task_name": "droid_sem_pos_dino",   "latent_name": "latents_pos_dino"},
}


def train_one(pos_key: str):
    cfg = POSITIONS[pos_key]
    task_name = cfg["task_name"]
    position = cfg["position"]

    config_path = STAMO_ROOT / "configs" / "droid.yaml"
    print(f"\n{'='*60}")
    print(f"Training position={position}, task={task_name}")
    print(f"{'='*60}")

    args = OmegaConf.load(str(config_path))
    args.deepspeed = False
    args.semantic_head.enabled = True
    args.semantic_head.lambda_weight = 1.0
    args.semantic_head.position = position
    args.task_name = task_name
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
            print(f"  [{task_name}] step {step}/{NUM_ITERS}  loss={avg_loss:.4f}  "
                  f"diff={avg_diff:.4f}  sem={avg_sem:.4f}  speed={speed:.1f} step/s")
            running_loss = running_diff = running_sem = 0.0
            t0 = time.time()

    ckpt_dir = STAMO_ROOT / args.train.ckpt_save_dir / task_name / str(NUM_ITERS)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(ckpt_dir), NUM_ITERS)
    print(f"  Saved checkpoint: {ckpt_dir}")

    del model, optimizer, scheduler, train_iter, dataset, train_loader
    gc.collect()
    torch.cuda.empty_cache()


def extract_features(pos_key: str):
    cfg = POSITIONS[pos_key]
    task_name = cfg["task_name"]
    position = cfg["position"]
    latent_name = cfg["latent_name"]

    config_path = STAMO_ROOT / "configs" / "droid.yaml"
    ckpt_dir = str(STAMO_ROOT / "ckpts" / task_name / str(NUM_ITERS))
    out_dir = ROOT / "data" / "interim" / "droid_process_chain" / latent_name

    args = OmegaConf.load(str(config_path))
    args.deepspeed = False
    args.semantic_head.enabled = True
    args.semantic_head.position = position

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
        print(f"  Extracting {len(rows)} features ({position}) from {manifest_path.name}...")

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


def run_eval(pos_key: str):
    cfg = POSITIONS[pos_key]
    position = cfg["position"]
    latent_name = cfg["latent_name"]

    latent_dir = ROOT / "data" / "interim" / "droid_process_chain" / latent_name
    train_manifest = ROOT / "data" / "processed" / "droid_process_chain" / "train.jsonl"
    eval_manifest = ROOT / "data" / "processed" / "droid_process_chain" / "test.jsonl"

    train_rows = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in eval_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

    def load_visual(row):
        data = np.load(latent_dir / f"{row['sample_id']}.npz")
        dz = data["delta_z"].reshape(-1).astype(np.float32)
        dp = data["delta_pooled"].reshape(-1).astype(np.float32)
        return np.concatenate([dz, dp])

    x_train = np.stack([load_visual(r) for r in train_rows])
    x_eval = np.stack([load_visual(r) for r in eval_rows])
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_eval_s = scaler.transform(x_eval)

    results = {}
    for clf_name, clf_cls in [
        ("HistGBT", lambda: HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.03, max_depth=6,
            min_samples_leaf=5, l2_regularization=0.1,
            class_weight="balanced", random_state=42)),
        ("LogReg", lambda: LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced",
            random_state=42, solver="lbfgs")),
    ]:
        print(f"\n  === Position {position} — {clf_name} (visual-only 2560D) ===")
        accs = {}
        for target in SEMANTIC_FIELDS:
            y_train = np.asarray([r["labels"][target] for r in train_rows])
            y_eval = np.asarray([r["labels"][target] for r in eval_rows])
            clf = clf_cls()
            clf.fit(x_train_s, y_train)
            acc = (clf.predict(x_eval_s) == y_eval).mean()
            accs[target] = float(acc)
            print(f"    {target}: {acc:.4f}")
        accs["mean"] = float(np.mean(list(accs.values())))
        print(f"    mean: {accs['mean']:.4f}")
        results[clf_name] = accs

    return results


def main():
    import subprocess

    all_results = {}

    for pos_key in ["pooled", "proj", "dino"]:
        latent_dir = ROOT / "data" / "interim" / "droid_process_chain" / POSITIONS[pos_key]["latent_name"]
        ckpt_dir = STAMO_ROOT / "ckpts" / POSITIONS[pos_key]["task_name"] / str(NUM_ITERS)

        # Check if we can skip training
        if latent_dir.exists() and len(list(latent_dir.glob("*.npz"))) > 100:
            print(f"\n[SKIP TRAIN+EXTRACT] {pos_key}: latents already exist at {latent_dir}")
        elif ckpt_dir.exists():
            print(f"\n[SKIP TRAIN] {pos_key}: checkpoint exists, extracting features...")
            # Run extract in subprocess to avoid CUDA state issues
            ret = subprocess.run(
                ["py", "-u", str(Path(__file__)), "--extract", pos_key],
                cwd=str(STAMO_ROOT),
            )
            if ret.returncode != 0:
                print(f"  [ERROR] extract failed for {pos_key}")
                continue
        else:
            print(f"\n--- Training + extracting {pos_key} (subprocess) ---")
            ret = subprocess.run(
                ["py", "-u", str(Path(__file__)), "--train-extract", pos_key],
                cwd=str(STAMO_ROOT),
            )
            if ret.returncode != 0:
                print(f"  [ERROR] train+extract failed for {pos_key}")
                continue

        print(f"\n--- Evaluating {pos_key} ---")
        all_results[pos_key] = run_eval(pos_key)

    # Print comparison table
    print(f"\n{'='*80}")
    print("POSITION ABLATION — Visual-Only (2560D) HistGBT")
    print(f"{'='*80}")
    fields = list(SEMANTIC_FIELDS)
    print(f"{'Position':<12}", end="")
    for f in fields:
        print(f"  {f:<18}", end="")
    print(f"  {'mean':>8}")
    print("-" * 80)

    for pos_key in ["pooled", "proj", "dino"]:
        res = all_results[pos_key]["HistGBT"]
        label = {"pooled": "A:pooled", "proj": "B:proj", "dino": "C:dino"}[pos_key]
        print(f"{label:<12}", end="")
        for f in fields:
            print(f"  {res[f]:<18.4f}", end="")
        print(f"  {res['mean']:>8.4f}")

    print(f"\n{'='*80}")
    print("POSITION ABLATION — Visual-Only (2560D) LogReg")
    print(f"{'='*80}")
    print(f"{'Position':<12}", end="")
    for f in fields:
        print(f"  {f:<18}", end="")
    print(f"  {'mean':>8}")
    print("-" * 80)

    for pos_key in ["pooled", "proj", "dino"]:
        res = all_results[pos_key]["LogReg"]
        label = {"pooled": "A:pooled", "proj": "B:proj", "dino": "C:dino"}[pos_key]
        print(f"{label:<12}", end="")
        for f in fields:
            print(f"  {res[f]:<18.4f}", end="")
        print(f"  {res['mean']:>8.4f}")

    # Save results
    out_path = ROOT / "results" / "ablation" / "head_position_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--train-extract":
        pos_key = sys.argv[2]
        print(f"\n--- Training {pos_key} ---")
        train_one(pos_key)
        print(f"\n--- Extracting features {pos_key} ---")
        extract_features(pos_key)
    elif len(sys.argv) >= 3 and sys.argv[1] == "--extract":
        pos_key = sys.argv[2]
        extract_features(pos_key)
    else:
        main()
