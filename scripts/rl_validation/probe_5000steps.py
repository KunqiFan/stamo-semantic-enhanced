"""Linear probe comparison at 5000 steps: Group A vs Group B.

Loads each checkpoint, encodes test set image pairs, computes delta_pooled,
then trains logistic regression on each semantic field.

Usage:
    cd "stamo_pro - 副本 - 副本"
    py rl_validation/scripts/probe_5000steps.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "StaMo"))

from stamo.renderer.model.renderer import RenderNet
from omegaconf import OmegaConf

SEMANTIC_FIELDS = ["contact_state", "gripper_state", "object_motion", "target_relation"]
TEST_MANIFEST = ROOT / "rl_validation" / "data" / "pickcube_v1" / "test.jsonl"
CONFIG_PATH = ROOT / "rl_validation" / "configs" / "stamo_maniskill.yaml"

CKPT_A = ROOT / "StaMo" / "ckpts" / "maniskill_pickcube_v1_diffonly" / "5000"
CKPT_B = ROOT / "StaMo" / "ckpts" / "maniskill_pickcube_v1_semantic" / "5000"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(ckpt_path: Path, cfg) -> RenderNet:
    model = RenderNet(cfg)
    model.load_checkpoint(str(ckpt_path))
    model.eval()
    model.to("cuda")
    return model


def load_test_data() -> list[dict]:
    rows = []
    with TEST_MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def extract_delta_pooled(model: RenderNet, rows: list[dict], batch_size: int = 16) -> np.ndarray:
    all_feats = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        imgs_t = []
        imgs_tp = []
        for row in batch:
            img_t = Image.open(row["image_t"]).convert("RGB")
            img_tp = Image.open(row["image_tp"]).convert("RGB")
            imgs_t.append(TRANSFORM(img_t))
            imgs_tp.append(TRANSFORM(img_tp))

        t_tensor = torch.stack(imgs_t).cuda()
        tp_tensor = torch.stack(imgs_tp).cuda()

        _, pooled_t = model.encode(t_tensor)
        _, pooled_tp = model.encode(tp_tensor)

        delta = (pooled_tp - pooled_t).cpu().float().numpy()
        all_feats.append(delta)

    return np.concatenate(all_feats, axis=0)


def run_probe(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray) -> float:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train_s, y_train)
    return clf.score(X_test_s, y_test)


def main():
    cfg = OmegaConf.load(str(CONFIG_PATH))

    rows = load_test_data()
    print(f"Test samples: {len(rows)}")

    # 70/30 split for train/test probe
    np.random.seed(42)
    indices = np.random.permutation(len(rows))
    split = int(0.7 * len(rows))
    train_idx, test_idx = indices[:split], indices[split:]
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    print(f"Probe train: {len(train_rows)}, Probe test: {len(test_rows)}")

    results = {}

    for group_name, ckpt_path in [("Group A (diff-only)", CKPT_A),
                                   ("Group B (diff+sem)", CKPT_B)]:
        print(f"\n{'='*50}")
        print(f"  {group_name}: {ckpt_path}")
        print(f"{'='*50}")

        if group_name == "Group B (diff+sem)":
            cfg.semantic_head.enabled = True
            cfg.semantic_head.lambda_weight = 0.1
        else:
            cfg.semantic_head.enabled = False
            cfg.semantic_head.lambda_weight = 0.0

        model = load_model(ckpt_path, cfg)

        print("Extracting features (train split)...")
        X_train = extract_delta_pooled(model, train_rows)
        print(f"  shape: {X_train.shape}")

        print("Extracting features (test split)...")
        X_test = extract_delta_pooled(model, test_rows)
        print(f"  shape: {X_test.shape}")

        del model
        torch.cuda.empty_cache()

        results[group_name] = {}
        for field in SEMANTIC_FIELDS:
            y_train = np.array([r["labels"][field] for r in train_rows])
            y_test = np.array([r["labels"][field] for r in test_rows])
            acc = run_probe(X_train, y_train, X_test, y_test)
            results[group_name][field] = acc
            print(f"  {field:20s}: {acc:.1%}")

        mean_acc = np.mean(list(results[group_name].values()))
        results[group_name]["MEAN"] = mean_acc
        print(f"  {'MEAN':20s}: {mean_acc:.1%}")

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY: Linear Probe Accuracy @ 5000 steps")
    print(f"{'='*70}")
    print(f"{'Field':20s} | {'Group A':12s} | {'Group B':12s} | {'Diff':8s}")
    print("-" * 60)
    for field in SEMANTIC_FIELDS + ["MEAN"]:
        a = results["Group A (diff-only)"][field]
        b = results["Group B (diff+sem)"][field]
        diff = b - a
        print(f"{field:20s} | {a:11.1%} | {b:11.1%} | {diff:+.1%}")

    # Save results
    out_path = ROOT / "rl_validation" / "results" / "probe_check_5000steps.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# StaMo ManiSkill3 训练对比检查 — 5000 步 checkpoint\n")
        f.write("# 日期: 2026-05-02\n")
        f.write("# 任务: PickCube-v1\n")
        f.write("# 评估方法: 线性探针 (logistic regression on delta_pooled, 70/30 split)\n\n")
        f.write("## 训练配置\n")
        f.write("- batch_size: 8, grad_accum: 1, effective_batch: 8\n")
        f.write("- lr: 5e-5 (cosine decay), num_workers: 4\n")
        f.write("- Group A: diffusion-only (semantic_head.enabled=false)\n")
        f.write("- Group B: diffusion+semantic (semantic_head.enabled=true, lambda=0.1)\n\n")
        f.write("## 训练 Loss (5000 步末)\n")
        f.write("- Group A: diffusion loss ≈ 0.20\n")
        f.write("- Group B: diffusion loss ≈ 0.20, semantic loss ≈ 0.29, total ≈ 0.23\n\n")
        f.write("## 线性探针准确率 (test set)\n\n")
        f.write(f"| 语义字段         | Group A (diff-only) | Group B (diff+sem) | 差异     |\n")
        f.write(f"|-----------------|--------------------|--------------------|----------|\n")
        for field in SEMANTIC_FIELDS:
            a = results["Group A (diff-only)"][field]
            b = results["Group B (diff+sem)"][field]
            diff = b - a
            f.write(f"| {field:15s} | {a:.1%}{'':14s} | {b:.1%}{'':14s} | {diff:+.1%}{'':4s} |\n")
        a_mean = results["Group A (diff-only)"]["MEAN"]
        b_mean = results["Group B (diff+sem)"]["MEAN"]
        diff_mean = b_mean - a_mean
        f.write(f"| **MEAN**        | **{a_mean:.1%}**{'':10s} | **{b_mean:.1%}**{'':10s} | **{diff_mean:+.1%}**|\n")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
