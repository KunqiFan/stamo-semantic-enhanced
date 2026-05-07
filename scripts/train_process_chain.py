from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MaxAbsScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stamo_bridge.semantics.interface import SEMANTIC_FIELDS, semantics_from_dict, semantics_vectorize


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_predicted_semantics(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                result[row["sample_id"]] = row["predicted_semantics"]
    return result


def semantics_matrix(rows: list[dict], predicted_semantics: dict[str, dict[str, str]] | None) -> np.ndarray:
    vectors = []
    for row in rows:
        fallback = {field: row["labels"][field] for field in SEMANTIC_FIELDS}
        values = predicted_semantics.get(row["sample_id"], fallback) if predicted_semantics is not None else fallback
        vectors.append(semantics_vectorize(semantics_from_dict(values)))
    return np.asarray(vectors, dtype=np.float32)


def text_inputs(rows: list[dict], text_key: str) -> list[str]:
    values = []
    for row in rows:
        text = row.get(text_key) or row.get("caption") or row.get("process_text") or ""
        values.append(text)
    return values


def _describe(name: str, mat) -> None:
    """Print shape and value range for diagnostic purposes."""
    if issparse(mat):
        dense_sample = mat[:min(5, mat.shape[0])].toarray()
    else:
        dense_sample = np.asarray(mat[:min(5, mat.shape[0])])
    print(f"  [{name}] shape={mat.shape}  min={dense_sample.min():.4f}  max={dense_sample.max():.4f}  mean={dense_sample.mean():.6f}")


def _run_stacking(
    text_train, text_eval, sem_train, sem_eval, y_train, y_eval,
) -> None:
    """Two-stage stacking: train separate classifiers, then combine probabilities."""
    print("\n  [stacking] Training text classifier ...")
    clf_text = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
    clf_text.fit(text_train, y_train)
    text_proba_train = clf_text.predict_proba(text_train)
    text_proba_eval = clf_text.predict_proba(text_eval)
    text_preds = clf_text.predict(text_eval)
    print(f"  [stacking] text-only sub-model:")
    print(classification_report(y_eval, text_preds, digits=4, zero_division=0))

    print("  [stacking] Training semantics classifier ...")
    # For semantics, HGB typically works better than LR on dense vectors
    clf_sem = HistGradientBoostingClassifier(max_iter=300, random_state=42)
    clf_sem.fit(sem_train.toarray() if issparse(sem_train) else sem_train, y_train)
    sem_proba_train = clf_sem.predict_proba(sem_train.toarray() if issparse(sem_train) else sem_train)
    sem_proba_eval = clf_sem.predict_proba(sem_eval.toarray() if issparse(sem_eval) else sem_eval)
    sem_preds = clf_sem.predict(sem_eval.toarray() if issparse(sem_eval) else sem_eval)
    print(f"  [stacking] sem-only sub-model:")
    print(classification_report(y_eval, sem_preds, digits=4, zero_division=0))

    # Meta-classifier on stacked probabilities
    meta_x_train = np.hstack([text_proba_train, sem_proba_train])
    meta_x_eval = np.hstack([text_proba_eval, sem_proba_eval])
    # Meta-classifier: LogisticRegression is usually fine, but let's use a small RandomForest for non-linear fusion
    meta_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    meta_clf.fit(meta_x_train, y_train)
    preds = meta_clf.predict(meta_x_eval)
    print("  [stacking] meta-classifier (text + sem combined):")
    print(classification_report(y_eval, preds, digits=4, zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--setting", choices=["text_only", "semantics_only", "text_plus_semantics"], required=True)
    parser.add_argument("--target", type=str, default="stage_label")
    parser.add_argument("--text_key", type=str, default="process_text")
    parser.add_argument("--predicted_semantics", type=Path, default=None)
    parser.add_argument("--fusion", choices=["concat", "stacking"], default="stacking",
                        help="Fusion strategy for text_plus_semantics: "
                             "'concat' = MaxAbsScaler + single LR; "
                             "'stacking' = separate classifiers + meta-LR on probabilities.")
    args = parser.parse_args()

    train_rows = load_rows(args.train_manifest)
    eval_rows = load_rows(args.eval_manifest)
    predicted = load_predicted_semantics(args.predicted_semantics) if args.predicted_semantics else None

    y_train = np.asarray([row["labels"][args.target] for row in train_rows])
    y_eval = np.asarray([row["labels"][args.target] for row in eval_rows])

    try:
        from sentence_transformers import SentenceTransformer
        print("\n  [info] Using sentence-transformers for text features")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        def get_text_features(texts, is_train):
            return encoder.encode(texts, show_progress_bar=False)
    except ImportError:
        print("\n  [info] sentence-transformers not found. Falling back to TF-IDF.")
        tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.95)
        def get_text_features(texts, is_train):
            if is_train:
                return tfidf.fit_transform(texts)
            return tfidf.transform(texts)

    if args.setting == "text_only":
        x_train = get_text_features(text_inputs(train_rows, args.text_key), True)
        x_eval = get_text_features(text_inputs(eval_rows, args.text_key), False)
        _describe("text", x_train)

        clf = LogisticRegression(max_iter=2000)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_eval)
        print(classification_report(y_eval, preds, digits=4, zero_division=0))

    elif args.setting == "semantics_only":
        x_train = csr_matrix(semantics_matrix(train_rows, predicted))
        x_eval = csr_matrix(semantics_matrix(eval_rows, predicted))
        _describe("semantics", x_train)

        clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
        clf.fit(x_train.toarray() if issparse(x_train) else x_train, y_train)
        preds = clf.predict(x_eval.toarray() if issparse(x_eval) else x_eval)
        print(classification_report(y_eval, preds, digits=4, zero_division=0))

    else:  # text_plus_semantics
        text_train = get_text_features(text_inputs(train_rows, args.text_key), True)
        text_eval = get_text_features(text_inputs(eval_rows, args.text_key), False)
        sem_train = csr_matrix(semantics_matrix(train_rows, predicted))
        sem_eval = csr_matrix(semantics_matrix(eval_rows, predicted))

        _describe("text", text_train)
        _describe("semantics", sem_train)

        if args.fusion == "stacking":
            _run_stacking(text_train, text_eval, sem_train, sem_eval, y_train, y_eval)
        else:
            # concat with MaxAbsScaler to normalise both feature groups
            x_train_raw = hstack([csr_matrix(text_train), sem_train]) if issparse(sem_train) else np.hstack([text_train, sem_train.toarray() if issparse(sem_train) else sem_train])
            x_eval_raw = hstack([csr_matrix(text_eval), sem_eval]) if issparse(sem_eval) else np.hstack([text_eval, sem_eval.toarray() if issparse(sem_eval) else sem_eval])
            scaler = MaxAbsScaler()
            x_train = scaler.fit_transform(x_train_raw)
            x_eval = scaler.transform(x_eval_raw)
            _describe("fused (scaled)", x_train)

            clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
            clf.fit(x_train.toarray() if issparse(x_train) else x_train, y_train)
            preds = clf.predict(x_eval.toarray() if issparse(x_eval) else x_eval)
            print(classification_report(y_eval, preds, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
