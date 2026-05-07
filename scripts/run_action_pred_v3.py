"""Action Prediction — same conditions as semantic interface experiment.

Predict ee_delta (6-dim) to test whether semantic interface helps action prediction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
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


def build_features(rows, condition, latent_dir, predicted):
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
        elif condition == "delta_z+gold":
            dz = load_latent(latent_dir, sid)
            g = gold_onehot_vector(row)
            vectors.append(np.concatenate([dz, g]))
        elif condition == "delta_z+hard":
            dz = load_latent(latent_dir, sid)
            h = hard_onehot_vector(sid, predicted)
            vectors.append(np.concatenate([dz, h]))
        elif condition == "delta_z+soft":
            dz = load_latent(latent_dir, sid)
            s = soft_probability_vector(sid, predicted)
            vectors.append(np.concatenate([dz, s]))
        else:
            raise ValueError(condition)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--latent_dir", type=Path, required=True)
    parser.add_argument("--predicted_train", type=Path, required=True)
    parser.add_argument("--predicted_eval", type=Path, required=True)
    parser.add_argument("--target", type=str, default="ee_delta")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--out_json", type=Path, default=ROOT / "results" / "action_pred_v3.json")
    args = parser.parse_args()

    train_rows = load_rows(args.train_manifest)
    eval_rows = load_rows(args.eval_manifest)
    pred_train = load_predicted_full(args.predicted_train)
    pred_eval = load_predicted_full(args.predicted_eval)

    y_train = np.array([r[args.target] for r in train_rows], dtype=np.float32)
    y_eval = np.array([r[args.target] for r in eval_rows], dtype=np.float32)

    print(f"Train: {len(train_rows)}, Eval: {len(eval_rows)}")
    print(f"Target: {args.target}, dim={y_train.shape[1]}\n")

    results = {}

    for condition in args.conditions:
        print(f"\n{'='*60}")
        print(f"  {condition}")
        print(f"{'='*60}")

        x_train = build_features(train_rows, condition, args.latent_dir, pred_train)
        x_eval = build_features(eval_rows, condition, args.latent_dir, pred_eval)
        print(f"  feature dim={x_train.shape[1]}")

        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    max_iter=500, learning_rate=0.05, max_depth=6,
                    min_samples_leaf=5, l2_regularization=0.1, random_state=42,
                ), n_jobs=-1
            )),
        ])

        reg.fit(x_train, y_train)
        preds = reg.predict(x_eval)

        mse = float(mean_squared_error(y_eval, preds))
        mae = float(mean_absolute_error(y_eval, preds))
        r2 = float(r2_score(y_eval, preds))

        dim_names = ["x", "y", "z", "rx", "ry", "rz"]
        per_dim = {dim_names[i]: round(float(r2_score(y_eval[:, i], preds[:, i])), 4)
                   for i in range(min(len(dim_names), y_eval.shape[1]))}

        print(f"  MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.4f}")
        print(f"  Per-dim R2: {per_dim}")

        results[condition] = {
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 4),
            "feature_dim": int(x_train.shape[1]),
            "per_dim_r2": per_dim,
        }

    print(f"\n{'='*60}")
    print(f"  SUMMARY ({args.target})")
    print(f"{'='*60}")
    print(f"{'Condition':<24} {'MSE':>10} {'MAE':>10} {'R2':>8} {'Dim':>6}")
    print("-" * 62)
    for cond, r in results.items():
        print(f"{cond:<24} {r['mse']:>10.6f} {r['mae']:>10.6f} {r['r2']:>8.4f} {r['feature_dim']:>6}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
