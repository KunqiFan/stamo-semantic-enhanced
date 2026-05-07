"""Ablation evaluation using ONLY visual features (delta_z + delta_pooled).

Removes proprioceptive signals and statistics to isolate Projector representation quality.
Also runs a weak classifier (Logistic Regression) alongside HistGBT to expose differences.

Usage:
    cd "stamo_pro - 副本 - 副本" && py -u scripts/eval_ablation_visual_only.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from stamo_bridge.semantics.interface import SEMANTIC_FIELDS

# All lambda latent directories (including 1.0 baseline)
LATENT_DIRS = {
    "0.1": ROOT / "data" / "interim" / "droid_process_chain" / "latents_sem_lam0.1",
    "0.5": ROOT / "data" / "interim" / "droid_process_chain" / "latents_sem_lam0.5",
    "1.0": ROOT / "data" / "interim" / "droid_process_chain" / "latents_sem_lam1.0",
    "2.0": ROOT / "data" / "interim" / "droid_process_chain" / "latents_sem_lam2.0",
    "5.0": ROOT / "data" / "interim" / "droid_process_chain" / "latents_sem_lam5.0",
}

TRAIN_MANIFEST = ROOT / "data" / "processed" / "droid_process_chain" / "train.jsonl"
EVAL_MANIFEST = ROOT / "data" / "processed" / "droid_process_chain" / "test.jsonl"


def load_visual_only(latent_dir: Path, row: dict) -> np.ndarray:
    """Load ONLY delta_z (2048D) + delta_pooled (512D) = 2560D."""
    data = np.load(latent_dir / f"{row['sample_id']}.npz")
    dz = data["delta_z"].reshape(-1).astype(np.float32)
    dp = data["delta_pooled"].reshape(-1).astype(np.float32)
    return np.concatenate([dz, dp])


def evaluate(latent_dir: Path, train_rows, eval_rows, classifier_name="hgbt"):
    """Run classification with specified classifier, return per-field accuracy."""
    x_train = np.stack([load_visual_only(latent_dir, r) for r in train_rows])
    x_eval = np.stack([load_visual_only(latent_dir, r) for r in eval_rows])

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_eval_s = scaler.transform(x_eval)

    results = {}
    for target in SEMANTIC_FIELDS:
        y_train = np.asarray([r["labels"][target] for r in train_rows])
        y_eval = np.asarray([r["labels"][target] for r in eval_rows])

        if classifier_name == "hgbt":
            clf = HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.03, max_depth=6,
                min_samples_leaf=5, l2_regularization=0.1,
                class_weight="balanced", random_state=42,
            )
        elif classifier_name == "logreg":
            clf = LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced",
                random_state=42, solver="lbfgs",
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        clf.fit(x_train_s, y_train)
        eval_acc = (clf.predict(x_eval_s) == y_eval).mean()
        results[target] = eval_acc

    return results


def print_table(title, all_results, fields):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(title)
    print(f"{'='*90}")
    print(f"{'lambda':<10}", end="")
    for f in fields:
        print(f"  {f:<18}", end="")
    print(f"  {'mean':>8}")
    print("-" * 90)

    for lam_str in sorted(all_results.keys(), key=float):
        res = all_results[lam_str]
        marker = "*" if lam_str == "1.0" else ""
        print(f"{lam_str + marker:<10}", end="")
        accs = []
        for f in fields:
            print(f"  {res[f]:<18.4f}", end="")
            accs.append(res[f])
        print(f"  {np.mean(accs):>8.4f}")


def main():
    train_rows = [json.loads(l) for l in TRAIN_MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in EVAL_MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Train: {len(train_rows)}, Eval: {len(eval_rows)}")

    fields = list(SEMANTIC_FIELDS)
    all_hgbt = {}
    all_logreg = {}

    for lam_str, latent_dir in sorted(LATENT_DIRS.items(), key=lambda x: float(x[0])):
        if not latent_dir.exists():
            print(f"  [SKIP] {lam_str}: {latent_dir} not found")
            continue

        print(f"\nEvaluating lambda={lam_str} (visual-only 2560D)...")
        print(f"  HistGradientBoosting:")
        hgbt_res = evaluate(latent_dir, train_rows, eval_rows, "hgbt")
        for f in fields:
            print(f"    {f}: {hgbt_res[f]:.4f}")
        all_hgbt[lam_str] = hgbt_res

        print(f"  Logistic Regression:")
        lr_res = evaluate(latent_dir, train_rows, eval_rows, "logreg")
        for f in fields:
            print(f"    {f}: {lr_res[f]:.4f}")
        all_logreg[lam_str] = lr_res

    # Print comparison tables
    print_table(
        "VISUAL-ONLY FEATURES (delta_z + delta_pooled = 2560D) — HistGradientBoosting",
        all_hgbt, fields
    )
    print_table(
        "VISUAL-ONLY FEATURES (delta_z + delta_pooled = 2560D) — Logistic Regression (LINEAR)",
        all_logreg, fields
    )

    # Compute spread for each table
    for name, results in [("HistGBT", all_hgbt), ("LogReg", all_logreg)]:
        means = [np.mean([v[f] for f in fields]) for v in results.values()]
        spread = max(means) - min(means)
        print(f"\n{name} mean accuracy spread: {spread*100:.2f}pp (min={min(means)*100:.2f}%, max={max(means)*100:.2f}%)")

    # Save results
    out_path = ROOT / "results" / "ablation" / "visual_only_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {"hgbt": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_hgbt.items()},
                 "logreg": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_logreg.items()}}
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
