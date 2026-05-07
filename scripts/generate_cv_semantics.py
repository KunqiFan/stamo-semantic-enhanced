"""Generate cross-validated predicted semantics for the TRAINING set,
so that train and test both have the same noise level in their semantics.
This eliminates the train/test distribution mismatch that kills downstream performance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stamo_bridge.semantics.interface import SEMANTIC_FIELDS, SEMANTIC_LABELS  # noqa: E402

# Reuse feature engineering from v2
sys.path.insert(0, str(ROOT / "scripts"))
from train_compact_semantics_v2 import load_features_extended  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--latent_dir", type=Path, required=True)
    parser.add_argument("--feature_key", default="extended")
    parser.add_argument("--out_train", type=Path, required=True,
                        help="Output: cross-validated predictions for TRAIN set")
    parser.add_argument("--out_eval", type=Path, required=True,
                        help="Output: predictions for EVAL/TEST set")
    args = parser.parse_args()

    train_rows = [json.loads(l) for l in args.train_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in args.eval_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"Train: {len(train_rows)}, Eval: {len(eval_rows)}")

    # Load features
    def load_feats(rows):
        return np.stack([
            load_features_extended(args.latent_dir, r["sample_id"])[args.feature_key]
            for r in rows
        ])

    x_train = load_feats(train_rows)
    x_eval = load_feats(eval_rows)

    # Standardize
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_eval_s = scaler.transform(x_eval)

    # Initialize output structures
    by_id_train: dict[str, dict] = {
        r["sample_id"]: {
            "sample_id": r["sample_id"],
            "gold_semantics": {},
            "predicted_semantics": {},
            "predicted_probabilities": {},
        } for r in train_rows
    }
    by_id_eval: dict[str, dict] = {
        r["sample_id"]: {
            "sample_id": r["sample_id"],
            "gold_semantics": {},
            "predicted_semantics": {},
            "predicted_probabilities": {},
        } for r in eval_rows
    }

    for target in SEMANTIC_FIELDS:
        y_train = np.asarray([r["labels"][target] for r in train_rows])
        y_eval = np.asarray([r["labels"][target] for r in eval_rows])

        clf = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.03, max_depth=6,
            min_samples_leaf=5, l2_regularization=0.1,
            class_weight="balanced", random_state=42,
        )

        # Cross-validated predictions for train set
        train_proba_cv = cross_val_predict(
            clf, x_train_s, y_train, cv=5, method="predict_proba", n_jobs=-1,
        )
        # Need to get class names — fit on full train
        clf.fit(x_train_s, y_train)
        classes = clf.classes_
        train_preds_cv = classes[np.argmax(train_proba_cv, axis=1)]

        # Full-train predictions for eval set
        eval_proba = clf.predict_proba(x_eval_s)
        eval_preds = clf.predict(x_eval_s)

        print(f"\n=== {target} (cross-val train) ===")
        from sklearn.metrics import classification_report
        print(classification_report(y_train, train_preds_cv, digits=4, zero_division=0))

        print(f"=== {target} (eval) ===")
        print(classification_report(y_eval, eval_preds, digits=4, zero_division=0))

        # Store train results
        for i, row in enumerate(train_rows):
            sid = row["sample_id"]
            by_id_train[sid]["gold_semantics"][target] = row["labels"][target]
            by_id_train[sid]["predicted_semantics"][target] = train_preds_cv[i]
            prob_dict = {cls: round(float(train_proba_cv[i, j]), 4) for j, cls in enumerate(classes)}
            by_id_train[sid]["predicted_probabilities"][target] = prob_dict

        # Store eval results
        for i, row in enumerate(eval_rows):
            sid = row["sample_id"]
            by_id_eval[sid]["gold_semantics"][target] = row["labels"][target]
            by_id_eval[sid]["predicted_semantics"][target] = eval_preds[i]
            prob_dict = {cls: round(float(eval_proba[i, j]), 4) for j, cls in enumerate(classes)}
            by_id_eval[sid]["predicted_probabilities"][target] = prob_dict

    # Write outputs
    for path, by_id in [(args.out_train, by_id_train), (args.out_eval, by_id_eval)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for sid in sorted(by_id):
                f.write(json.dumps(by_id[sid]) + "\n")
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
