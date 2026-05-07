from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stamo_bridge.semantics.interface import SEMANTIC_FIELDS

# ---------------------------------------------------------------------------
# Classifier zoo
# ---------------------------------------------------------------------------

CLASSIFIER_REGISTRY: dict[str, Pipeline] = {
    "logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ]),
    "mlp": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=False,
            alpha=0.01,
            random_state=42,
        )),
    ]),
    "pca_mlp": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=64)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=False,
            alpha=0.01,
            random_state=42,
        )),
    ]),
    "pca_logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=64)),
        ("clf", LogisticRegression(max_iter=2000)),
    ]),
    "rf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)),
    ]),
    "hgb": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)),
    ]),
}


def _make_classifier(name: str) -> Pipeline:
    """Return a fresh clone of the named classifier pipeline."""
    from sklearn.base import clone
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Unknown classifier '{name}'. Choose from: {list(CLASSIFIER_REGISTRY)}")
    return clone(CLASSIFIER_REGISTRY[name])


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
    if feature_key == "both":
        dz = data["delta_z"].reshape(-1)
        dp = data["delta_pooled"].reshape(-1)
        return np.concatenate([dz, dp]).astype(np.float32)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--latent_dir", type=Path, required=True)
    parser.add_argument("--feature_key", choices=["delta_z", "delta_pooled", "both", "pca_z_plus_pooled"], default="delta_z")
    parser.add_argument("--classifier", choices=list(CLASSIFIER_REGISTRY), default="logistic",
                        help="Classifier head to use. "
                             "'logistic' = StandardScaler+LR (baseline probe); "
                             "'mlp' = StandardScaler+MLP(128,64); "
                             "'pca_mlp' = StandardScaler+PCA(64)+MLP(64,32); "
                             "'pca_logistic' = StandardScaler+PCA(64)+LR; "
                             "'rf' = StandardScaler+RandomForest; "
                             "'hgb' = StandardScaler+HistGradientBoosting.")
    parser.add_argument("--out_path", type=Path, required=True)
    args = parser.parse_args()

    train_rows = load_rows(args.train_manifest)
    eval_rows = load_rows(args.eval_manifest)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Feature key: {args.feature_key}")
    print(f"Classifier:  {args.classifier}")
    print(f"Train size:  {len(train_rows)}")
    print(f"Eval size:   {len(eval_rows)}")

    # Pre-calculate for pca_z_plus_pooled
    if args.feature_key == "pca_z_plus_pooled":
        x_train_dz = np.stack([load_features(args.latent_dir, row["sample_id"], "delta_z") for row in train_rows])
        x_eval_dz = np.stack([load_features(args.latent_dir, row["sample_id"], "delta_z") for row in eval_rows])
        x_train_dp = np.stack([load_features(args.latent_dir, row["sample_id"], "delta_pooled") for row in train_rows])
        x_eval_dp = np.stack([load_features(args.latent_dir, row["sample_id"], "delta_pooled") for row in eval_rows])
        
        pca = PCA(n_components=64)
        x_train_dz_pca = pca.fit_transform(x_train_dz)
        x_eval_dz_pca = pca.transform(x_eval_dz)
        
        precomputed_x_train = np.concatenate([x_train_dz_pca, x_train_dp], axis=1)
        precomputed_x_eval = np.concatenate([x_eval_dz_pca, x_eval_dp], axis=1)
        print(f"Feature dim: {precomputed_x_train.shape[1]} (64 PCA + 128 pooled)")
    else:
        sample_feat = load_features(args.latent_dir, train_rows[0]["sample_id"], args.feature_key)
        print(f"Feature dim: {sample_feat.shape[0]}")

    prediction_rows: list[dict] = []

    for target in SEMANTIC_FIELDS:
        if args.feature_key == "pca_z_plus_pooled":
            x_train = precomputed_x_train
            x_eval = precomputed_x_eval
            y_train = np.asarray([row["labels"][target] for row in train_rows])
            y_eval = np.asarray([row["labels"][target] for row in eval_rows])
        else:
            x_train, y_train = build_xy(train_rows, args.latent_dir, args.feature_key, target)
            x_eval, y_eval = build_xy(eval_rows, args.latent_dir, args.feature_key, target)
        unique_train = np.unique(y_train)

        if len(unique_train) < 2:
            preds = np.repeat(unique_train[0], len(y_eval))
            print(f"\n=== {target} ===")
            print(f"Train split has a single class: {unique_train[0]}. Using constant predictor.")
        else:
            clf = _make_classifier(args.classifier)

            # Disable early_stopping when dataset is too small for reliable validation split
            clf_step = clf.named_steps.get("clf", None)
            if clf_step is not None and getattr(clf_step, "early_stopping", False):
                if len(x_train) < 50 or len(unique_train) < 3:
                    clf_step.early_stopping = False

            clf.fit(x_train, y_train)
            preds = clf.predict(x_eval)
            print(f"\n=== {target} ===")

        print(classification_report(y_eval, preds, digits=4, zero_division=0))

        for row, pred in zip(eval_rows, preds):
            prediction_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "target": target,
                    "gold": row["labels"][target],
                    "pred": pred,
                }
            )

    by_id: dict[str, dict] = {}
    for row in prediction_rows:
        entry = by_id.setdefault(row["sample_id"], {"sample_id": row["sample_id"], "gold_semantics": {}, "predicted_semantics": {}})
        entry["gold_semantics"][row["target"]] = row["gold"]
        entry["predicted_semantics"][row["target"]] = row["pred"]

    with args.out_path.open("w", encoding="utf-8") as f:
        for sample_id in sorted(by_id):
            f.write(json.dumps(by_id[sample_id]) + "\n")

    print(f"\nSaved predicted compact semantics to {args.out_path}")


if __name__ == "__main__":
    main()
