"""Robustness Experiment — compare representation forms under noisy predictions.

With v3 predicted semantics (~60% all-4-correct), compare how different
representation forms transmit semantic information to stage classification:

  1. raw_delta_z         — baseline, no semantic interface
  2. hard_onehot         — argmax one-hot from predicted semantics (12-dim)
  3. soft_probability    — full probability distribution (12-dim)
  4. template_text       — SBERT encoding of template text (384-dim)
  5. enriched_text       — SBERT encoding of enriched text (384-dim)
  6. delta_z + hard      — fusion: delta_z + hard one-hot
  7. delta_z + soft      — fusion: delta_z + soft probability
  8. delta_z + template  — fusion: delta_z + template text embedding
  9. delta_z + enriched  — fusion: delta_z + enriched text embedding

All predict stage_label (gold) from predicted (noisy) semantics.
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

from stamo_bridge.semantics.interface import SEMANTIC_FIELDS, SEMANTIC_LABELS


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_predicted_full(path: Path) -> dict[str, dict]:
    """Load predicted semantics with both labels and probabilities."""
    result = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                result[row["sample_id"]] = row
    return result


def load_latent(latent_dir: Path, sample_id: str) -> np.ndarray:
    data = np.load(latent_dir / f"{sample_id}.npz")
    return data["delta_z"].reshape(-1).astype(np.float32)


def load_text_embedding(embed_dir: Path, sample_id: str, field: str) -> np.ndarray:
    data = np.load(embed_dir / f"{sample_id}.npz")
    return data[field].astype(np.float32)


def gold_onehot_vector(row: dict) -> np.ndarray:
    vec = []
    for field in SEMANTIC_FIELDS:
        gold_val = row["labels"][field]
        for lbl in SEMANTIC_LABELS[field]:
            vec.append(1.0 if lbl == gold_val else 0.0)
    return np.array(vec, dtype=np.float32)


def hard_onehot_vector(sid: str, predicted: dict) -> np.ndarray:
    entry = predicted[sid]
    pred_labels = entry["predicted_semantics"]
    vec = []
    for field in SEMANTIC_FIELDS:
        pred_val = pred_labels[field]
        for lbl in SEMANTIC_LABELS[field]:
            vec.append(1.0 if lbl == pred_val else 0.0)
    return np.array(vec, dtype=np.float32)


def soft_probability_vector(sid: str, predicted: dict) -> np.ndarray:
    entry = predicted[sid]
    probs = entry["predicted_probabilities"]
    vec = []
    for field in SEMANTIC_FIELDS:
        for lbl in SEMANTIC_LABELS[field]:
            vec.append(probs[field].get(lbl, 0.0))
    return np.array(vec, dtype=np.float32)


def build_features(rows, condition, latent_dir, embed_dir, predicted):
    vectors = []
    for row in rows:
        sid = row["sample_id"]
        if condition == "raw_delta_z":
            vectors.append(load_latent(latent_dir, sid))
        elif condition == "gold_onehot":
            vectors.append(gold_onehot_vector(row))
        elif condition == "hard_onehot":
            vectors.append(hard_onehot_vector(sid, predicted))
        elif condition == "soft_probability":
            vectors.append(soft_probability_vector(sid, predicted))
        elif condition == "template_text":
            vectors.append(load_text_embedding(embed_dir, sid, "template_text"))
        elif condition == "enriched_text":
            vectors.append(load_text_embedding(embed_dir, sid, "enriched_text"))
        elif condition == "delta_z+hard":
            dz = load_latent(latent_dir, sid)
            h = hard_onehot_vector(sid, predicted)
            vectors.append(np.concatenate([dz, h]))
        elif condition == "delta_z+soft":
            dz = load_latent(latent_dir, sid)
            s = soft_probability_vector(sid, predicted)
            vectors.append(np.concatenate([dz, s]))
        elif condition == "delta_z+template":
            dz = load_latent(latent_dir, sid)
            t = load_text_embedding(embed_dir, sid, "template_text")
            vectors.append(np.concatenate([dz, t]))
        elif condition == "delta_z+enriched":
            dz = load_latent(latent_dir, sid)
            e = load_text_embedding(embed_dir, sid, "enriched_text")
            vectors.append(np.concatenate([dz, e]))
        elif condition == "delta_z+gold":
            dz = load_latent(latent_dir, sid)
            g = gold_onehot_vector(row)
            vectors.append(np.concatenate([dz, g]))
        else:
            raise ValueError(f"Unknown condition: {condition}")
    return np.stack(vectors)


CONDITIONS = [
    "raw_delta_z",
    "gold_onehot",
    "hard_onehot",
    "soft_probability",
    "delta_z+gold",
    "delta_z+hard",
    "delta_z+soft",
]


def make_classifier(condition: str) -> Pipeline:
    if "delta_z" in condition and "+" not in condition:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500, alpha=0.005, random_state=42,
            )),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=6,
            min_samples_leaf=5, l2_regularization=0.1,
            class_weight="balanced", random_state=42,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--latent_dir", type=Path, required=True)
    parser.add_argument("--text_embed_dir", type=Path, required=True)
    parser.add_argument("--predicted_train", type=Path, required=True)
    parser.add_argument("--predicted_eval", type=Path, required=True)
    parser.add_argument("--target", type=str, default="stage_label")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--out_json", type=Path, default=ROOT / "results" / "robustness_experiment.json")
    args = parser.parse_args()

    train_rows = load_rows(args.train_manifest)
    eval_rows = load_rows(args.eval_manifest)
    pred_train = load_predicted_full(args.predicted_train)
    pred_eval = load_predicted_full(args.predicted_eval)

    y_train = np.array([r["labels"][args.target] for r in train_rows])
    y_eval = np.array([r["labels"][args.target] for r in eval_rows])

    print(f"Train: {len(train_rows)}, Eval: {len(eval_rows)}")
    print(f"Target: {args.target}")
    print(f"Classes: {sorted(set(y_train))}\n")

    results = {}

    for condition in args.conditions:
        print(f"\n{'='*60}")
        print(f"  {condition}")
        print(f"{'='*60}")

        x_train = build_features(train_rows, condition, args.latent_dir, args.text_embed_dir, pred_train)
        x_eval = build_features(eval_rows, condition, args.latent_dir, args.text_embed_dir, pred_eval)
        print(f"  dim={x_train.shape[1]}")

        clf = make_classifier(condition)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_eval)

        macro_f1 = f1_score(y_eval, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_eval, preds, average="weighted", zero_division=0)
        accuracy = float(np.mean(preds == y_eval))

        print(classification_report(y_eval, preds, digits=4, zero_division=0))

        results[condition] = {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "feature_dim": int(x_train.shape[1]),
        }

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<24} {'Acc':>8} {'Macro-F1':>10} {'W-F1':>10} {'Dim':>6}")
    print("-" * 62)
    for cond, r in results.items():
        print(f"{cond:<24} {r['accuracy']:>8.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>10.4f} {r['feature_dim']:>6}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
