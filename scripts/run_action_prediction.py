"""Text Bridge Action Prediction Experiment — regression task.

Predict next-step ee_delta (6-dim) or action (7-dim) from various
semantic representations. This is a harder task than stage classification
because action prediction requires understanding the continuous dynamics
of manipulation, not just discrete stage labels.

Conditions:
  A. raw_delta_z           — raw visual latent difference
  B. discrete_labels       — one-hot compact semantics
  C. template_text         — SBERT encoding of template text
  D. enriched_text         — SBERT encoding of enriched text
  + fusion variants

Metrics: MSE, MAE, R² on ee_delta prediction.
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

from stamo_bridge.semantics.interface import (
    SEMANTIC_FIELDS,
    semantics_from_dict,
    semantics_vectorize,
)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_latent(latent_dir: Path, sample_id: str) -> np.ndarray:
    data = np.load(latent_dir / f"{sample_id}.npz")
    return data["delta_z"].reshape(-1).astype(np.float32)


def load_text_embedding(embed_dir: Path, sample_id: str, field: str) -> np.ndarray:
    data = np.load(embed_dir / f"{sample_id}.npz")
    return data[field].astype(np.float32)


def discrete_labels_vector(row: dict, predicted=None) -> np.ndarray:
    sid = row["sample_id"]
    if predicted and sid in predicted:
        values = {f: predicted[sid].get(f, row["labels"][f]) for f in SEMANTIC_FIELDS}
    else:
        values = {f: row["labels"][f] for f in SEMANTIC_FIELDS}
    return semantics_vectorize(semantics_from_dict(values))


def load_predicted_semantics(path: Path) -> dict[str, dict[str, str]]:
    result = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                result[row["sample_id"]] = row.get("predicted_semantics", {})
    return result


def build_features(rows, condition, latent_dir, embed_dir, predicted=None):
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
        elif condition == "discrete_plus_enriched":
            dl = discrete_labels_vector(row, predicted)
            te = load_text_embedding(embed_dir, sid, "enriched_text")
            vectors.append(np.concatenate([dl, te]))
        elif condition == "delta_z_plus_discrete":
            dz = load_latent(latent_dir, sid)
            dl = discrete_labels_vector(row, predicted)
            vectors.append(np.concatenate([dz, dl]))
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


def make_regressor():
    base = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=5,
        l2_regularization=0.1,
        random_state=42,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", MultiOutputRegressor(base, n_jobs=-1)),
    ])


CONDITIONS = [
    "raw_delta_z",
    "discrete_labels",
    "template_text",
    "enriched_text",
    "discrete_plus_enriched",
    "delta_z_plus_discrete",
    "delta_z_plus_enriched",
    "full_fusion",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--latent_dir", type=Path, required=True)
    parser.add_argument("--text_embed_dir", type=Path, required=True)
    parser.add_argument("--predicted_semantics_train", type=Path, default=None)
    parser.add_argument("--predicted_semantics_eval", type=Path, default=None)
    parser.add_argument("--target", type=str, default="ee_delta",
                        choices=["ee_delta", "action"])
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--out_json", type=Path, default=ROOT / "results" / "action_prediction_experiment.json")
    args = parser.parse_args()

    train_rows = load_rows(args.train_manifest)
    eval_rows = load_rows(args.eval_manifest)

    pred_train = load_predicted_semantics(args.predicted_semantics_train) if args.predicted_semantics_train else None
    pred_eval = load_predicted_semantics(args.predicted_semantics_eval) if args.predicted_semantics_eval else None

    y_train = np.array([r[args.target] for r in train_rows], dtype=np.float32)
    y_eval = np.array([r[args.target] for r in eval_rows], dtype=np.float32)

    print(f"Train: {len(train_rows)}, Eval: {len(eval_rows)}")
    print(f"Target: {args.target} (dim={y_train.shape[1]})")
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

        reg = make_regressor()
        reg.fit(x_train, y_train)
        preds = reg.predict(x_eval)

        mse = float(mean_squared_error(y_eval, preds))
        mae = float(mean_absolute_error(y_eval, preds))
        r2 = float(r2_score(y_eval, preds))

        per_dim_r2 = [float(r2_score(y_eval[:, i], preds[:, i])) for i in range(y_eval.shape[1])]

        print(f"  MSE: {mse:.6f}  MAE: {mae:.6f}  R2: {r2:.4f}")
        print(f"  Per-dim R2: {[f'{v:.4f}' for v in per_dim_r2]}")

        results[condition] = {
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 4),
            "per_dim_r2": [round(v, 4) for v in per_dim_r2],
            "feature_dim": int(x_train.shape[1]),
        }

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({args.target} prediction)")
    print(f"{'='*60}")
    print(f"{'Condition':<28} {'MSE':>10} {'MAE':>10} {'R2':>8} {'Dim':>6}")
    print("-" * 66)
    for cond, r in results.items():
        print(f"{cond:<28} {r['mse']:>10.6f} {r['mae']:>10.6f} {r['r2']:>8.4f} {r['feature_dim']:>6}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out_json}")


if __name__ == "__main__":
    main()
