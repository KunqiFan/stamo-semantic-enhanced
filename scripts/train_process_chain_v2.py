"""Process chain classifier v2 — uses soft probability vectors from
compact semantics (not hard one-hot), injects raw ee_delta physical
features, and applies multi-level fusion strategies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stamo_bridge.semantics.interface import SEMANTIC_FIELDS, SEMANTIC_LABELS  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_predicted_semantics_v2(path: Path) -> dict[str, dict]:
    """Load v2 predictions that include predicted_probabilities."""
    result: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                result[row["sample_id"]] = row
    return result


def soft_semantics_vector(
    sample_id: str,
    predicted: dict[str, dict] | None,
    row_labels: dict,
) -> np.ndarray:
    """Build a soft probability vector for all semantic fields.

    If predicted_probabilities are available (v2 format), use them.
    Otherwise fall back to one-hot from gold or hard predictions.
    """
    vector = []
    for field in SEMANTIC_FIELDS:
        labels = SEMANTIC_LABELS[field]
        if predicted and sample_id in predicted:
            entry = predicted[sample_id]
            # Try soft probabilities first
            if "predicted_probabilities" in entry and field in entry["predicted_probabilities"]:
                probs = entry["predicted_probabilities"][field]
                for lbl in labels:
                    vector.append(probs.get(lbl, 0.0))
                continue
            # Fall back to hard prediction
            if "predicted_semantics" in entry and field in entry["predicted_semantics"]:
                pred_val = entry["predicted_semantics"][field]
                for lbl in labels:
                    vector.append(1.0 if lbl == pred_val else 0.0)
                continue
        # Gold labels (one-hot)
        gold_val = row_labels[field]
        for lbl in labels:
            vector.append(1.0 if lbl == gold_val else 0.0)
    return np.array(vector, dtype=np.float32)


def get_ee_delta(row: dict) -> np.ndarray:
    """Extract ee_delta as a physical feature vector."""
    ee = row.get("ee_delta")
    if ee is None:
        return np.zeros(6, dtype=np.float32)
    arr = np.array(ee, dtype=np.float32)
    # Also compute derived features
    xyz_norm = float(np.linalg.norm(arr[:3]))
    z_component = arr[2] if len(arr) > 2 else 0.0
    rot_norm = float(np.linalg.norm(arr[3:6])) if len(arr) > 3 else 0.0
    derived = np.array([xyz_norm, z_component, rot_norm], dtype=np.float32)
    return np.concatenate([arr, derived])


def text_inputs(rows: list[dict], text_key: str) -> list[str]:
    values = []
    for row in rows:
        text = row.get(text_key) or row.get("caption") or row.get("process_text") or ""
        values.append(text)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--setting", choices=[
        "text_only", "semantics_only", "text_plus_semantics",
        "physics_only", "semantics_plus_physics",
        "all_modalities",
    ], required=True)
    parser.add_argument("--target", type=str, default="stage_label")
    parser.add_argument("--text_key", type=str, default="caption")
    parser.add_argument("--predicted_semantics", type=Path, default=None,
                        help="Predicted semantics for EVAL/TEST set")
    parser.add_argument("--predicted_semantics_train", type=Path, default=None,
                        help="Cross-validated predicted semantics for TRAIN set (critical for fair comparison)")
    args = parser.parse_args()

    train_rows = load_rows(args.train_manifest)
    eval_rows = load_rows(args.eval_manifest)
    predicted_eval = load_predicted_semantics_v2(args.predicted_semantics) if args.predicted_semantics else None
    predicted_train = load_predicted_semantics_v2(args.predicted_semantics_train) if args.predicted_semantics_train else None

    y_train = np.asarray([row["labels"][args.target] for row in train_rows])
    y_eval = np.asarray([row["labels"][args.target] for row in eval_rows])

    # --- Build text features ---
    try:
        from sentence_transformers import SentenceTransformer
        print("\n  [info] Using sentence-transformers for text features")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        def encode_text(texts):
            return encoder.encode(texts, show_progress_bar=False)
        text_train = encode_text(text_inputs(train_rows, args.text_key))
        text_eval = encode_text(text_inputs(eval_rows, args.text_key))
    except ImportError:
        print("\n  [info] sentence-transformers not found. Falling back to TF-IDF.")
        tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.95,
                                sublinear_tf=True)
        text_train = tfidf.fit_transform(text_inputs(train_rows, args.text_key))
        text_eval = tfidf.transform(text_inputs(eval_rows, args.text_key))

    # --- Build soft semantics features ---
    sem_train = np.stack([
        soft_semantics_vector(r["sample_id"], predicted_train, r["labels"])
        for r in train_rows
    ])
    sem_eval = np.stack([
        soft_semantics_vector(r["sample_id"], predicted_eval, r["labels"])
        for r in eval_rows
    ])

    # --- Build physics features (ee_delta) ---
    phys_train = np.stack([get_ee_delta(r) for r in train_rows])
    phys_eval = np.stack([get_ee_delta(r) for r in eval_rows])

    def to_dense(x):
        return x.toarray() if issparse(x) else np.asarray(x)

    # --- Prepare result printer ---
    def train_and_report(name: str, x_tr, x_ev):
        x_tr = to_dense(x_tr)
        x_ev = to_dense(x_ev)
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr)
        x_ev = scaler.transform(x_ev)

        print(f"\n  [{name}] Feature dim: {x_tr.shape[1]}")

        # Use HGB as the primary classifier with balanced weights
        clf = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            max_depth=6,
            min_samples_leaf=5,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(x_tr, y_train)
        preds = clf.predict(x_ev)
        print(classification_report(y_eval, preds, digits=4, zero_division=0))
        return preds

    # ---- Settings ----
    if args.setting == "text_only":
        train_and_report("text_only", text_train, text_eval)

    elif args.setting == "semantics_only":
        train_and_report("semantics_only", sem_train, sem_eval)

    elif args.setting == "physics_only":
        train_and_report("physics_only", phys_train, phys_eval)

    elif args.setting == "semantics_plus_physics":
        # Dense features only: soft semantics + ee_delta physics
        x_tr = np.hstack([sem_train, phys_train])
        x_ev = np.hstack([sem_eval, phys_eval])
        train_and_report("semantics_plus_physics", x_tr, x_ev)

    elif args.setting == "text_plus_semantics":
        # Dense features: soft semantics + ee_delta physics (no sparse text drowning)
        x_tr = np.hstack([sem_train, phys_train])
        x_ev = np.hstack([sem_eval, phys_eval])
        print("\n  === Semantics + Physics (dense) baseline ===")
        train_and_report("sem_plus_phys", x_tr, x_ev)

        # Now add text via proper cross-validated stacking
        from sklearn.model_selection import cross_val_predict
        print("\n  === Cross-validated Stacking (Text + Semantics + Physics) ===")

        # Build a text-only HGB to get cross-validated probabilities on train set
        text_tr_dense = to_dense(text_train)
        text_ev_dense = to_dense(text_eval)
        scaler_text = StandardScaler()
        text_tr_scaled = scaler_text.fit_transform(text_tr_dense)
        text_ev_scaled = scaler_text.transform(text_ev_dense)

        text_clf = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.03, max_depth=6,
            min_samples_leaf=5, l2_regularization=0.1,
            class_weight="balanced", random_state=42,
        )
        # Cross-val predict for training set (avoid overfitting)
        text_proba_train_cv = cross_val_predict(
            text_clf, text_tr_scaled, y_train, cv=5, method="predict_proba", n_jobs=-1
        )
        # Refit on full train for eval set
        text_clf.fit(text_tr_scaled, y_train)
        text_proba_eval = text_clf.predict_proba(text_ev_scaled)

        # Dense features sub-model: also cross-val
        dense_tr = np.hstack([sem_train, phys_train])
        dense_ev = np.hstack([sem_eval, phys_eval])
        scaler_dense = StandardScaler()
        dense_tr_scaled = scaler_dense.fit_transform(dense_tr)
        dense_ev_scaled = scaler_dense.transform(dense_ev)

        dense_clf = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.03, max_depth=6,
            min_samples_leaf=5, l2_regularization=0.1,
            class_weight="balanced", random_state=42,
        )
        dense_proba_train_cv = cross_val_predict(
            dense_clf, dense_tr_scaled, y_train, cv=5, method="predict_proba", n_jobs=-1
        )
        dense_clf.fit(dense_tr_scaled, y_train)
        dense_proba_eval = dense_clf.predict_proba(dense_ev_scaled)

        # Meta-classifier on stacked cross-val probabilities + raw soft semantics
        meta_train = np.hstack([text_proba_train_cv, dense_proba_train_cv, sem_train, phys_train])
        meta_eval = np.hstack([text_proba_eval, dense_proba_eval, sem_eval, phys_eval])

        scaler_meta = StandardScaler()
        meta_train = scaler_meta.fit_transform(meta_train)
        meta_eval = scaler_meta.transform(meta_eval)

        print(f"\n  [meta-classifier] Feature dim: {meta_train.shape[1]}")

        meta_clf = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            max_depth=5,
            min_samples_leaf=5,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=42,
        )
        meta_clf.fit(meta_train, y_train)
        meta_preds = meta_clf.predict(meta_eval)
        print("\n  === FINAL: Meta-classifier (Text + Soft Semantics + Physics) ===")
        print(classification_report(y_eval, meta_preds, digits=4, zero_division=0))

    elif args.setting == "all_modalities":
        # Dense concatenation only (sem + phys), no sparse text
        x_tr = np.hstack([sem_train, phys_train])
        x_ev = np.hstack([sem_eval, phys_eval])
        train_and_report("all_modalities_dense", x_tr, x_ev)


if __name__ == "__main__":
    main()
