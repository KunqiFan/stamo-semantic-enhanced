"""Compact semantics classifier v2 — outputs soft probabilities, uses
stronger features, handles class imbalance, and exports both hard
predictions and full probability vectors for downstream fusion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stamo_bridge.semantics.interface import SEMANTIC_FIELDS  # noqa: E402

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def load_features_extended(latent_dir: Path, sample_id: str) -> dict[str, np.ndarray]:
    """Load latents and compute engineered features."""
    data = np.load(latent_dir / f"{sample_id}.npz")
    dz = data["delta_z"].reshape(-1).astype(np.float32)
    dp = data["delta_pooled"].reshape(-1).astype(np.float32)

    # --- Handcrafted statistics over delta_z ---
    dz_2d = data["delta_z"].astype(np.float32)  # shape (2, 256)
    stats = []
    for row in dz_2d:
        stats.extend([
            row.mean(), row.std(), np.abs(row).max(),
            np.percentile(row, 25), np.percentile(row, 75),
            float(np.linalg.norm(row)),
            float((row > 0).sum()) / len(row),   # fraction positive
        ])
    stats_arr = np.array(stats, dtype=np.float32)

    # --- Cross features: element-wise product of pooled ---
    # Top-k activations of delta_z (reduces noise)
    topk = 32
    abs_dz = np.abs(dz)
    topk_idx = np.argsort(abs_dz)[-topk:]
    topk_feats = dz[topk_idx]

    return {
        "delta_z": dz,
        "delta_pooled": dp,
        "both": np.concatenate([dz, dp]),
        "extended": np.concatenate([dz, dp, stats_arr, topk_feats]),
        "stats": stats_arr,
    }


def build_xy(
    rows: list[dict],
    latent_dir: Path,
    feature_key: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    feats_list = []
    labels = []
    for row in rows:
        feats = load_features_extended(latent_dir, row["sample_id"])
        feats_list.append(feats[feature_key])
        labels.append(row["labels"][target])
    return np.stack(feats_list), np.asarray(labels)


# ---------------------------------------------------------------------------
# Classifier zoo v2 — stronger defaults, class-weight balanced
# ---------------------------------------------------------------------------

def make_classifier(name: str) -> Pipeline:
    """Build a fresh classifier pipeline."""
    registry = {
        "hgb": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.03,
                max_depth=6,
                min_samples_leaf=5,
                l2_regularization=0.1,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                max_depth=25,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "gb": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42,
            )),
        ]),
        "stacking": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", StackingClassifier(
                estimators=[
                    ("hgb", HistGradientBoostingClassifier(
                        max_iter=300, learning_rate=0.05, max_depth=5,
                        class_weight="balanced", random_state=42)),
                    ("rf", RandomForestClassifier(
                        n_estimators=200, max_depth=20,
                        class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
                    ("lr", LogisticRegression(
                        max_iter=2000, C=0.5, class_weight="balanced")),
                ],
                final_estimator=LogisticRegression(max_iter=2000, C=1.0),
                cv=3,
                passthrough=False,
                n_jobs=-1,
            )),
        ]),
        "mlp_deep": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=128)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.15,
                alpha=0.005,
                learning_rate="adaptive",
                random_state=42,
            )),
        ]),
    }
    if name not in registry:
        raise ValueError(f"Unknown classifier '{name}'. Choose from: {list(registry)}")
    from sklearn.base import clone
    return clone(registry[name])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--latent_dir", type=Path, required=True)
    parser.add_argument("--feature_key", choices=["delta_z", "delta_pooled", "both", "extended", "stats"], default="extended")
    parser.add_argument("--classifier", choices=["hgb", "rf", "gb", "stacking", "mlp_deep"], default="hgb")
    parser.add_argument("--out_path", type=Path, required=True)
    args = parser.parse_args()

    train_rows = []
    with args.train_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_rows.append(json.loads(line))
    eval_rows = []
    with args.eval_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_rows.append(json.loads(line))

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Feature key: {args.feature_key}")
    print(f"Classifier:  {args.classifier}")
    print(f"Train size:  {len(train_rows)}")
    print(f"Eval size:   {len(eval_rows)}")

    # Build prediction structures
    by_id: dict[str, dict] = {}
    for row in eval_rows:
        by_id[row["sample_id"]] = {
            "sample_id": row["sample_id"],
            "gold_semantics": {},
            "predicted_semantics": {},
            "predicted_probabilities": {},  # NEW: soft probabilities per field
        }

    for target in SEMANTIC_FIELDS:
        x_train, y_train = build_xy(train_rows, args.latent_dir, args.feature_key, target)
        x_eval, y_eval = build_xy(eval_rows, args.latent_dir, args.feature_key, target)

        unique_train = np.unique(y_train)
        if len(unique_train) < 2:
            preds = np.repeat(unique_train[0], len(y_eval))
            print(f"\n=== {target} ===")
            print(f"Single class in train: {unique_train[0]}")
            proba = np.zeros((len(y_eval), 1))
            proba[:, 0] = 1.0
            classes = unique_train
        else:
            clf = make_classifier(args.classifier)

            # Disable early_stopping if dataset too small
            clf_step = clf.named_steps.get("clf", None)
            if clf_step is not None and getattr(clf_step, "early_stopping", False):
                if len(x_train) < 100 or len(unique_train) < 3:
                    clf_step.early_stopping = False

            clf.fit(x_train, y_train)
            preds = clf.predict(x_eval)
            proba = clf.predict_proba(x_eval)
            classes = clf.classes_

            print(f"\n=== {target} ===")
            print(classification_report(y_eval, preds, digits=4, zero_division=0))

        # Store results
        for i, row in enumerate(eval_rows):
            sid = row["sample_id"]
            by_id[sid]["gold_semantics"][target] = row["labels"][target]
            by_id[sid]["predicted_semantics"][target] = preds[i]
            # Store probability dict for this field
            prob_dict = {}
            for j, cls_name in enumerate(classes):
                prob_dict[cls_name] = round(float(proba[i, j]), 4)
            by_id[sid]["predicted_probabilities"][target] = prob_dict

    # Write output
    with args.out_path.open("w", encoding="utf-8") as f:
        for sample_id in sorted(by_id):
            f.write(json.dumps(by_id[sample_id]) + "\n")

    print(f"\nSaved predicted compact semantics (with probabilities) to {args.out_path}")


if __name__ == "__main__":
    main()
