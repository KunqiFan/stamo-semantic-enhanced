"""Lambda ablation experiment for semantic-parallel training.

Trains 5000 steps for each lambda value, then extracts delta_z and runs
sklearn classification to compare semantic classification accuracy.

Usage:
    cd StaMo && py ../scripts/run_lambda_ablation.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STAMO_ROOT = ROOT / "StaMo"
if str(STAMO_ROOT) not in sys.path:
    sys.path.insert(0, str(STAMO_ROOT))

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from stamo.renderer.model.renderer import RenderNet
from stamo.renderer.utils.data import PairImageData, collate_fn_pair
from stamo.renderer.utils.optim import (
    WarmupLinearConstantLR,
    WarmupLinearLR,
    get_criterion,
    get_optimizer,
)

# Add src to path for sklearn evaluation
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


LAMBDA_VALUES = [0.1, 0.5, 2.0, 5.0]
NUM_ITERS = 5000
GRAD_ACCUM = 4


def vram_gb():
    return torch.cuda.memory_allocated() / 1024**3


def train_one_lambda(lam: float, config_path: Path):
    """Train semantic-parallel model with given lambda for 5000 steps."""
    task_name = f"droid_sem_lam{lam}"
    print(f"\n{'='*60}")
    print(f"Training lambda={lam}, task={task_name}")
    print(f"{'='*60}")

    args = OmegaConf.load(str(config_path))
    args.deepspeed = False
    args.semantic_head.enabled = True
    args.semantic_head.lambda_weight = lam
    args.task_name = task_name
    args.train.num_iters = NUM_ITERS
    args.train.save_step = NUM_ITERS  # only save final

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
    torch.cuda.reset_peak_memory_stats()

    running_loss = running_diff = running_sem = 0.0
    t0 = time.time()
    loss_history = []

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
            loss_history.append({
                "step": step, "total": avg_loss, "diff": avg_diff, "sem": avg_sem
            })
            running_loss = running_diff = running_sem = 0.0
            t0 = time.time()

    # Save checkpoint
    ckpt_dir = STAMO_ROOT / args.train.ckpt_save_dir / task_name / str(NUM_ITERS)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(ckpt_dir), NUM_ITERS)
    print(f"  Saved checkpoint: {ckpt_dir}")

    # Cleanup
    del model, optimizer, scheduler, train_iter, dataset, train_loader
    gc.collect()
    torch.cuda.empty_cache()

    return loss_history


def extract_delta_z(config_path: Path, ckpt_dir: str, out_dir: Path, manifest: Path):
    """Extract delta_z features from a checkpoint."""
    from stamo.renderer.model.renderer import RenderNet
    import torchvision.transforms as T
    from PIL import Image

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
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

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
    print(f"  Extracted {len(rows)} features to {out_dir}")


def run_sklearn_eval(latent_dir: Path, train_manifest: Path, eval_manifest: Path):
    """Run sklearn HistGradientBoosting classification and return eval accuracies."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from stamo_bridge.semantics.interface import SEMANTIC_FIELDS

    train_rows = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in eval_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

    def load_features(row):
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

    x_train = np.stack([load_features(r) for r in train_rows])
    x_eval = np.stack([load_features(r) for r in eval_rows])

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_eval_s = scaler.transform(x_eval)

    results = {}
    for target in SEMANTIC_FIELDS:
        y_train = np.asarray([r["labels"][target] for r in train_rows])
        y_eval = np.asarray([r["labels"][target] for r in eval_rows])

        clf = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.03, max_depth=6,
            min_samples_leaf=5, l2_regularization=0.1,
            class_weight="balanced", random_state=42,
        )
        clf.fit(x_train_s, y_train)
        eval_acc = (clf.predict(x_eval_s) == y_eval).mean()
        results[target] = eval_acc
        print(f"    {target}: {eval_acc:.4f}")

    return results


def main():
    config_path = STAMO_ROOT / "configs" / "droid.yaml"
    train_manifest = ROOT / "data" / "processed" / "droid_process_chain" / "train.jsonl"
    eval_manifest = ROOT / "data" / "processed" / "droid_process_chain" / "test.jsonl"

    all_results = {}

    for lam in LAMBDA_VALUES:
        tag = f"lam{lam}"
        latent_dir = ROOT / "data" / "interim" / "droid_process_chain" / f"latents_sem_{tag}"

        # Step 1: Train
        loss_history = train_one_lambda(lam, config_path)

        # Step 2: Extract features
        ckpt_dir = str(STAMO_ROOT / "ckpts" / f"droid_sem_{tag}" / str(NUM_ITERS))
        if not Path(ckpt_dir).exists():
            # task_name in training uses different format
            ckpt_dir = str(STAMO_ROOT / "ckpts" / task_name / str(NUM_ITERS))
        print(f"\nExtracting features (lambda={lam})...")
        for manifest in [train_manifest, eval_manifest]:
            extract_delta_z(config_path, ckpt_dir, latent_dir, manifest)

        # Step 3: Evaluate
        print(f"\nSklearn eval (lambda={lam})...")
        eval_results = run_sklearn_eval(latent_dir, train_manifest, eval_manifest)
        eval_results["loss_history"] = loss_history
        all_results[tag] = eval_results

    # Final comparison table
    print(f"\n{'='*70}")
    print("LAMBDA ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"{'lambda':<10}", end="")
    fields = ["contact_state", "gripper_state", "object_motion", "target_relation"]
    for f in fields:
        print(f"  {f:<18}", end="")
    print(f"  {'mean':>8}")
    print("-" * 100)

    # Include baseline lambda=1.0
    print(f"{'1.0*':<10}", end="")
    baseline = {"contact_state": 0.959, "gripper_state": 0.915,
                "object_motion": 0.994, "target_relation": 0.646}
    for f in fields:
        print(f"  {baseline[f]:<18.4f}", end="")
    print(f"  {np.mean(list(baseline.values())):>8.4f}")

    for tag, results in all_results.items():
        lam_str = tag.replace("lam", "")
        print(f"{lam_str:<10}", end="")
        accs = []
        for f in fields:
            print(f"  {results[f]:<18.4f}", end="")
            accs.append(results[f])
        print(f"  {np.mean(accs):>8.4f}")

    # Save results
    out_path = ROOT / "data" / "interim" / "droid_process_chain" / "lambda_ablation_results.json"
    with open(out_path, "w") as f:
        # Convert non-serializable items
        save_results = {}
        for k, v in all_results.items():
            save_results[k] = {kk: vv for kk, vv in v.items() if kk != "loss_history"}
            save_results[k]["loss_history"] = v.get("loss_history", [])
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
