"""Text Bridge Few-Shot Experiment — test whether text bridge generalizes
better than discrete labels under low-data regimes.

Core hypothesis: When training data is scarce, Sentence-BERT embeddings of
process descriptions carry pre-trained semantic structure that helps
generalization, while one-hot discrete labels have no such prior.

This script subsamples the training set at various fractions (5%, 10%, 20%,
50%, 100%) and compares all conditions at each level. We use GOLD labels
for all conditions to isolate the representation effect from upstream noise.

Usage:
    py scripts/run_fewshot_experiment.py \
        --train_manifest data/interim/droid_process_chain/bridge_texts/train.jsonl \
        --eval_manifest data/interim/droid_process_chain/bridge_texts/test.jsonl \
        --latent_dir data/interim/droid_process_chain/latents \
        --text_embed_dir data/interim/droid_process_chain/text_embeddings \
        --out_json results/fewshot_experiment.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
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


def discrete_labels_vector(row: dict) -> np.ndarray:
    values = {f: row["labels"][f] for f in SEMANTIC_FIELDS}
    return semantics_vectorize(semantics_from_dict(values))


FEATURE_BUILDERS = {
    "raw_delta_z": lambda row, ld, ed: load_latent(ld, row["sample_id"]),
    "discrete_labels": lambda row, ld, ed: discrete_labels_vector(row),
    "template_text": lambda row, ld, ed: load_text_embedding(ed, row["sample_id"], "template_text"),
    "enriched_text": lambda row, ld, ed: load_text_embedding(ed, row["sample_id"], "enriched_text"),
    "discrete_plus_enriched": lambda row, ld, ed: np.concatenate([
        discrete_labels_vector(row),
        load_text_embedding(ed, row["sample_id"], "enriched_text"),
    ]),
    "delta_z_plus_enriched": lambda row, ld, ed: np.concatenate([
        load_latent(ld, row["sample_id"]),
        load_text_embedding(ed, row["sample_id"], "enriched_text"),
    ]),
}


def build_features(rows, condition, latent_dir, embed_dir):
    builder = FEATURE_BUILDERS[condition]
    return np.stack([builder(r, latent_dir, embed_dir) for r in rows])


def make_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=3,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--latent_dir", type=Path, required=True)
    parser.add_argument("--text_embed_dir", type=Path, required=True)
    parser.add_argument("--target", type=str, default="stage_label")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.50, 1.0])
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--conditions", nargs="+", default=list(FEATURE_BUILDERS.keys()))
    parser.add_argument("--out_json", type=Path, default=ROOT / "results" / "fewshot_experiment.json")
    args = parser.parse_args()

    train_rows = load_rows(args.train_manifest)
    eval_rows = load_rows(args.eval_manifest)

    y_train_all = np.array([r["labels"][args.target] for r in train_rows])
    y_eval = np.array([r["labels"][args.target] for r in eval_rows])

    print(f"Train: {len(train_rows)}, Eval: {len(eval_rows)}")
    print(f"Target: {args.target}, Classes: {sorted(set(y_train_all))}")
    print(f"Fractions: {args.fractions}, Repeats: {args.n_repeats}")
    print(f"Conditions: {args.conditions}\n")

    # Pre-build all eval features
    eval_features = {}
    for cond in args.conditions:
        eval_features[cond] = build_features(eval_rows, cond, args.latent_dir, args.text_embed_dir)

    # Pre-build all train features
    train_features = {}
    for cond in args.conditions:
        train_features[cond] = build_features(train_rows, cond, args.latent_dir, args.text_embed_dir)

    results = {}

    for frac in args.fractions:
        print(f"\n{'='*60}")
        print(f"  Fraction: {frac:.0%} ({int(len(train_rows) * frac)} samples)")
        print(f"{'='*60}")

        frac_results = {}

        for cond in args.conditions:
            x_eval = eval_features[cond]
            x_train_full = train_features[cond]

            scores = []

            if frac >= 1.0:
                clf = make_classifier()
                clf.fit(x_train_full, y_train_all)
                preds = clf.predict(x_eval)
                scores.append(f1_score(y_eval, preds, average="macro", zero_division=0))
            else:
                splitter = StratifiedShuffleSplit(
                    n_splits=args.n_repeats, train_size=frac, random_state=42
                )
                for train_idx, _ in splitter.split(x_train_full, y_train_all):
                    x_sub = x_train_full[train_idx]
                    y_sub = y_train_all[train_idx]

                    clf = make_classifier()
                    clf.fit(x_sub, y_sub)
                    preds = clf.predict(x_eval)
                    scores.append(f1_score(y_eval, preds, average="macro", zero_division=0))

            mean_f1 = float(np.mean(scores))
            std_f1 = float(np.std(scores))
            print(f"  {cond:<28} macro-F1: {mean_f1:.4f} +/- {std_f1:.4f}")

            frac_results[cond] = {
                "mean_macro_f1": round(mean_f1, 4),
                "std_macro_f1": round(std_f1, 4),
                "n_train": int(len(train_rows) * frac),
                "scores": [round(s, 4) for s in scores],
            }

        results[f"{frac:.2f}"] = frac_results

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY (Macro-F1)")
    print(f"{'='*70}")
    header = f"{'Condition':<28}"
    for frac in args.fractions:
        header += f" {frac:>7.0%}"
    print(header)
    print("-" * (28 + 8 * len(args.fractions)))

    for cond in args.conditions:
        line = f"{cond:<28}"
        for frac in args.fractions:
            r = results[f"{frac:.2f}"][cond]
            line += f" {r['mean_macro_f1']:>7.4f}"
        print(line)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out_json}")


if __name__ == "__main__":
    main()
