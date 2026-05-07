"""Text Bridge Experiment — four-condition comparison for stage_label classification.

Conditions:
  A. raw_delta_z           — MLP on raw delta_z latent
  B. discrete_labels       — one-hot from predicted compact semantics
  C. template_text         — Sentence-BERT encoding of template text (from predicted semantics)
  D. enriched_text         — Sentence-BERT encoding of enriched text (from predicted semantics)
  + fusion variants

All conditions predict stage_label using GOLD labels as target.
Conditions B/C/D use PREDICTED semantics as input features (not gold).

Usage:
    py scripts/run_text_bridge_experiment.py \
        --train_manifest data/interim/droid_process_chain/bridge_texts_predicted/train.jsonl \
        --eval_manifest data/interim/droid_process_chain/bridge_texts_predicted/test.jsonl \
        --latent_dir data/interim/droid_process_chain/latents \
        --text_embed_dir data/interim/droid_process_chain/text_embeddings_predicted \
        --predicted_semantics_train data/interim/droid_process_chain/semantics/train_cv_predicted.jsonl \
        --predicted_semantics_eval data/interim/droid_process_chain/semantics/test_cv_predicted.jsonl \
        --out_json results/droid_text_bridge_predicted_no_leak.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stamo_bridge.semantics.interface import (
    SEMANTIC_FIELDS,
    SEMANTIC_LABELS,
    semantics_from_dict,
    semantics_vectorize,
)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_predicted_semantics(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                result[row["sample_id"]] = row.get("predicted_semantics", {})
    return result


def load_latent(latent_dir: Path, sample_id: str, key: str = "delta_z") -> np.ndarray:
    data = np.load(latent_dir / f"{sample_id}.npz")
    return data[key].reshape(-1).astype(np.float32)


def load_text_embedding(embed_dir: Path, sample_id: str, field: str) -> np.ndarray:
    data = np.load(embed_dir / f"{sample_id}.npz")
    return data[field].astype(np.float32)


def discrete_labels_vector(row: dict, predicted: dict[str, dict[str, str]] | None) -> np.ndarray:
    sid = row["sample_id"]
    if predicted and sid in predicted:
        values = {f: predicted[sid].get(f, row["labels"][f]) for f in SEMANTIC_FIELDS}
    else:
        values = {f: row["labels"][f] for f in SEMANTIC_FIELDS}
    return semantics_vectorize(semantics_from_dict(values))


def build_features(
    rows: list[dict],
    condition: str,
    latent_dir: Path | None,
    embed_dir: Path | None,
    predicted: dict[str, dict[str, str]] | None,
) -> np.ndarray:
    vectors = []
    for row in rows:
        sid = row["sample_id"]
        if condition == "raw_delta_z":
            vectors.append(load_latent(latent_dir, sid))
        elif condition == "discrete_labels":
            vectors.append(discrete_labels_vector(row, predicted))
        elif condition == "template_text":
            vectors.append(load_text_embedding(embed_dir, sid, "template_text"))
        elif condition == "enriched_text":
            vectors.append(load_text_embedding(embed_dir, sid, "enriched_text"))
        elif condition == "discrete_plus_template":
            dl = discrete_labels_vector(row, predicted)
            te = load_text_embedding(embed_dir, sid, "template_text")
            vectors.append(np.concatenate([dl, te]))
        elif condition == "discrete_plus_enriched":
            dl = discrete_labels_vector(row, predicted)
            te = load_text_embedding(embed_dir, sid, "enriched_text")
            vectors.append(np.concatenate([dl, te]))
        elif condition == "delta_z_plus_enriched":
            dz = load_latent(latent_dir, sid)
            te = load_text_embedding(embed_dir, sid, "enriched_text")
            vectors.append(np.concatenate([dz, te]))
        elif condition == "full_fusion":
            dz = load_latent(latent_dir, sid)
            dl = discrete_labels_vector(row, predicted)
            te = load_text_embedding(embed_dir, sid, "enriched_text")
            vectors.append(np.concatenate([dz, dl, te]))
        else:
            raise ValueError(f"Unknown condition: {condition}")
    return np.stack(vectors)


def make_classifier(condition: str) -> Pipeline:
    if condition == "raw_delta_z":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                early_stopping=False,
                alpha=0.005,
                random_state=42,
            )),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=5,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=42,
        )),
    ])


CONDITIONS = [
    "raw_delta_z",
    "discrete_labels",
    "template_text",
    "enriched_text",
    "discrete_plus_template",
    "discrete_plus_enriched",
    "delta_z_plus_enriched",
    "full_fusion",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--latent_dir", type=Path, required=True)
    parser.add_argument("--text_embed_dir", type=Path, required=True)
    parser.add_argument("--predicted_semantics_train", type=Path, default=None,
                        help="Predicted semantics for train set (for discrete_labels condition)")
    parser.add_argument("--predicted_semantics_eval", type=Path, default=None,
                        help="Predicted semantics for eval set (for discrete_labels condition)")
    parser.add_argument("--target", type=str, default="stage_label")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--out_json", type=Path, default=ROOT / "results" / "text_bridge_experiment.json")
    args = parser.parse_args()

    train_rows = load_rows(args.train_manifest)
    eval_rows = load_rows(args.eval_manifest)

    pred_train = load_predicted_semantics(args.predicted_semantics_train) if args.predicted_semantics_train else None
    pred_eval = load_predicted_semantics(args.predicted_semantics_eval) if args.predicted_semantics_eval else None

    y_train = np.array([r["labels"][args.target] for r in train_rows])
    y_eval = np.array([r["labels"][args.target] for r in eval_rows])

    print(f"Train: {len(train_rows)}, Eval: {len(eval_rows)}")
    print(f"Target: {args.target}")
    print(f"Classes: {sorted(set(y_train))}")
    print(f"Using predicted semantics: {pred_train is not None}\n")

    results = {}

    for condition in args.conditions:
        print(f"\n{'='*60}")
        print(f"  Condition: {condition}")
        print(f"{'='*60}")

        try:
            x_train = build_features(train_rows, condition, args.latent_dir, args.text_embed_dir, pred_train)
            x_eval = build_features(eval_rows, condition, args.latent_dir, args.text_embed_dir, pred_eval)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            continue

        print(f"  Feature dim: {x_train.shape[1]}")

        clf = make_classifier(condition)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_eval)

        report = classification_report(y_eval, preds, digits=4, zero_division=0)
        macro_f1 = f1_score(y_eval, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_eval, preds, average="weighted", zero_division=0)
        accuracy = float(np.mean(preds == y_eval))

        print(report)

        results[condition] = {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "feature_dim": int(x_train.shape[1]),
            "report": classification_report(y_eval, preds, digits=4, zero_division=0, output_dict=True),
        }

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<28} {'Acc':>8} {'Macro-F1':>10} {'W-F1':>10} {'Dim':>6}")
    print("-" * 66)
    for cond, r in results.items():
        print(f"{cond:<28} {r['accuracy']:>8.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>10.4f} {r['feature_dim']:>6}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out_json}")


if __name__ == "__main__":
    main()
