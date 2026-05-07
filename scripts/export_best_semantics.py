"""Export best predicted semantics using per-attribute optimal classifiers.

Based on DROID 100 Round 2 results:
- contact_state:  PCA-MLP + delta_pooled  (acc 0.651)
- gripper_state:  PCA-MLP + delta_pooled  (acc 0.619)
- object_motion:  MLP + delta_pooled      (acc 0.746)
- target_relation: PCA-MLP + delta_pooled (acc 0.444)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stamo_bridge.semantics.interface import SEMANTIC_FIELDS

# ---------------------------------------------------------------------------
# Per-attribute optimal configurations
# ---------------------------------------------------------------------------

BEST_CONFIG: dict[str, dict] = {
    "contact_state": {
        "feature_key": "both",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)),
        ]),
    },
    "gripper_state": {
        "feature_key": "both",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)),
        ]),
    },
    "object_motion": {
        "feature_key": "both",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)),
        ]),
    },
    "target_relation": {
        "feature_key": "both",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)),
        ]),
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_features(latent_dir: Path, sample_id: str, feature_key: str) -> np.ndarray:
    data = np.load(latent_dir / f"{sample_id}.npz")
    feature = data[feature_key]
    return feature.reshape(-1).astype(np.float32)


def build_xy(rows: list[dict], latent_dir: Path, feature_key: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([load_features(latent_dir, row["sample_id"], feature_key) for row in rows])
    y = np.asarray([row["labels"][target] for row in rows])
    return x, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export best predicted semantics using per-attribute optimal classifiers.")
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--latent_dir", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    args = parser.parse_args()

    train_rows = load_rows(args.train_manifest)
    eval_rows = load_rows(args.eval_manifest)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Train size: {len(train_rows)}")
    print(f"Eval size:  {len(eval_rows)}")
    print("Using per-attribute optimal classifiers")

    # Pre-calculate for pca_z_plus_pooled
    x_train_dz = np.stack([load_features(args.latent_dir, row["sample_id"], "delta_z") for row in train_rows])
    x_eval_dz = np.stack([load_features(args.latent_dir, row["sample_id"], "delta_z") for row in eval_rows])
    x_train_dp = np.stack([load_features(args.latent_dir, row["sample_id"], "delta_pooled") for row in train_rows])
    x_eval_dp = np.stack([load_features(args.latent_dir, row["sample_id"], "delta_pooled") for row in eval_rows])
    
    pca = PCA(n_components=64)
    x_train_dz_pca = pca.fit_transform(x_train_dz)
    x_eval_dz_pca = pca.transform(x_eval_dz)
    
    x_train_pca_both = np.concatenate([x_train_dz_pca, x_train_dp], axis=1)
    x_eval_pca_both = np.concatenate([x_eval_dz_pca, x_eval_dp], axis=1)

    prediction_rows: list[dict] = []

    for target in SEMANTIC_FIELDS:
        config = BEST_CONFIG[target]
        feature_key = config["feature_key"]
        clf = clone(config["pipeline"])

        if feature_key == "pca_z_plus_pooled":
            x_train = x_train_pca_both
            x_eval = x_eval_pca_both
            y_train = np.asarray([row["labels"][target] for row in train_rows])
            y_eval = np.asarray([row["labels"][target] for row in eval_rows])
        else:
            x_train, y_train = build_xy(train_rows, args.latent_dir, feature_key, target)
            x_eval, y_eval = build_xy(eval_rows, args.latent_dir, feature_key, target)

        unique_train = np.unique(y_train)
        if len(unique_train) < 2:
            preds = np.repeat(unique_train[0], len(y_eval))
            print(f"\n=== {target} (feature={feature_key}) ===")
            print(f"Single class: {unique_train[0]}. Constant predictor.")
        else:
            clf.fit(x_train, y_train)
            preds = clf.predict(x_eval)
            print(f"\n=== {target} (feature={feature_key}, pipeline={type(clf.named_steps.get('clf', clf)).__name__}) ===")

        print(classification_report(y_eval, preds, digits=4, zero_division=0))

        for row, pred in zip(eval_rows, preds):
            prediction_rows.append({
                "sample_id": row["sample_id"],
                "target": target,
                "gold": row["labels"][target],
                "pred": pred,
            })

    # Aggregate by sample_id
    by_id: dict[str, dict] = {}
    for row in prediction_rows:
        entry = by_id.setdefault(row["sample_id"], {
            "sample_id": row["sample_id"],
            "gold_semantics": {},
            "predicted_semantics": {},
        })
        entry["gold_semantics"][row["target"]] = row["gold"]
        entry["predicted_semantics"][row["target"]] = row["pred"]

    with args.out_path.open("w", encoding="utf-8") as f:
        for sample_id in sorted(by_id):
            f.write(json.dumps(by_id[sample_id]) + "\n")

    print(f"\nSaved best predicted semantics to {args.out_path}")


if __name__ == "__main__":
    main()
