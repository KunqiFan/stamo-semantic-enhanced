"""Quick semantic classification accuracy evaluation on a checkpoint.

Runs the SemanticHead forward pass on the test set and reports per-field accuracy.
No sklearn needed: uses the model's own classification heads directly.

Usage:
    cd StaMo && py ../scripts/eval_semantic_quick.py --ckpt ckpts/droid_semantic/7000
    cd StaMo && py ../scripts/eval_semantic_quick.py --ckpt ckpts/droid_semantic/9000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAMO_ROOT = ROOT / "StaMo"
if str(STAMO_ROOT) not in sys.path:
    sys.path.insert(0, str(STAMO_ROOT))

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from stamo.renderer.model.renderer import RenderNet
from stamo.renderer.utils.data import PairImageData, collate_fn_pair


LABEL_NAMES = {
    "contact_state": ["no_contact", "contact"],
    "gripper_state": ["open", "closing", "closed"],
    "object_motion": ["still", "lifted", "moved", "placed"],
    "target_relation": ["farther", "closer", "reached"],
}


@torch.no_grad()
def evaluate_semantic(model, dataloader):
    all_preds = {f: [] for f in LABEL_NAMES}
    all_golds = {f: [] for f in LABEL_NAMES}
    total_sem_loss = 0.0
    n_batches = 0

    ce_fn = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        images_t = batch["images_t"].to("cuda")
        images_tp = batch["images_tp"].to("cuda")
        labels = {k: v.to("cuda") for k, v in batch["labels"].items()}
        bsz = images_t.shape[0]

        stacked = torch.cat([images_t, images_tp], dim=0)
        proj_stacked = model.projector_feature_extractor(stacked)
        _, all_pooled = model.encode(proj_stacked)
        pooled_t, pooled_tp = all_pooled.chunk(2, dim=0)
        delta_pooled = pooled_tp - pooled_t

        logits = model.semantic_head(delta_pooled.float())

        batch_loss = 0.0
        for field, logit in logits.items():
            pred = logit.argmax(dim=-1).cpu().numpy()
            gold = labels[field].cpu().numpy()
            all_preds[field].extend(pred.tolist())
            all_golds[field].extend(gold.tolist())
            batch_loss += ce_fn(logit, labels[field]).item()
        total_sem_loss += batch_loss / len(logits)
        n_batches += 1

    results = {}
    for field in LABEL_NAMES:
        preds = np.array(all_preds[field])
        golds = np.array(all_golds[field])
        acc = (preds == golds).mean()
        results[field] = {
            "accuracy": acc,
            "n_samples": len(golds),
        }
        # Per-class accuracy
        classes = sorted(set(golds.tolist()))
        per_class = {}
        for c in classes:
            mask = golds == c
            if mask.sum() > 0:
                per_class[c] = (preds[mask] == c).mean()
        results[field]["per_class_accuracy"] = per_class

    results["avg_semantic_loss"] = total_sem_loss / max(n_batches, 1)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=STAMO_ROOT / "configs" / "droid.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (relative to StaMo/)")
    parser.add_argument("--batch_size", type=int, default=4)
    cli = parser.parse_args()

    ckpt_path = str(STAMO_ROOT / cli.ckpt) if not Path(cli.ckpt).is_absolute() else cli.ckpt

    args = OmegaConf.load(str(cli.config))
    args.deepspeed = False
    args.semantic_head.enabled = True

    model = RenderNet(args)
    model = model.to("cuda")
    step = model.load_checkpoint(ckpt_path)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path} (step {step})")

    dataset = PairImageData(
        args.data.eval_manifest, flip_p=0.0, img_size=args.data.img_size,
    )
    dataloader = DataLoader(
        dataset, batch_size=cli.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn_pair, drop_last=False,
    )
    print(f"Test set: {len(dataset)} samples")

    results = evaluate_semantic(model, dataloader)

    print(f"\n{'='*60}")
    print(f"Semantic Classification @ step {step}")
    print(f"{'='*60}")
    print(f"Avg semantic loss: {results['avg_semantic_loss']:.4f}")
    print(f"\n{'Field':<20} {'Accuracy':>10}")
    print("-" * 32)
    accs = []
    for field in LABEL_NAMES:
        r = results[field]
        print(f"{field:<20} {r['accuracy']:>10.4f}")
        accs.append(r["accuracy"])
        # Per-class details
        for cls_idx, cls_acc in r["per_class_accuracy"].items():
            cls_name = LABEL_NAMES[field][cls_idx] if cls_idx < len(LABEL_NAMES[field]) else str(cls_idx)
            print(f"  {cls_name:<18} {cls_acc:>10.4f}")
    print("-" * 32)
    print(f"{'Mean accuracy':<20} {np.mean(accs):>10.4f}")


if __name__ == "__main__":
    main()
