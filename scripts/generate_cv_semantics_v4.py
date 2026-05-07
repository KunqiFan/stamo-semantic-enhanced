"""Generate cross-validated predicted semantics using DINOv2 features + proprioceptive signals.

v4: Replaces delta_z/delta_pooled with DINOv2 CLS-token features (feat_t, feat_delta),
which showed dramatically better semantic classification (e.g. contact_state 80% vs 54%).

Usage:
    py scripts/generate_cv_semantics_v4.py \
        --train_manifest data/processed/droid_process_chain/train.jsonl \
        --eval_manifest  data/processed/droid_process_chain/test.jsonl \
        --dinov2_dir     data/interim/droid_process_chain/dinov2_features \
        --latent_dir     data/interim/droid_process_chain/latents_droid5k \
        --out_train      data/interim/droid_process_chain/semantics/train_v4.jsonl \
        --out_eval       data/interim/droid_process_chain/semantics/eval_v4.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stamo_bridge.semantics.interface import SEMANTIC_FIELDS  # noqa: E402


def load_features(dinov2_dir: Path, latent_dir: Path | None, row: dict) -> np.ndarray:
    sid = row["sample_id"]

    # DINOv2 features (768D each)
    dino = np.load(dinov2_dir / f"{sid}.npz")
    feat_t = dino["feat_t"].reshape(-1).astype(np.float32)
    feat_delta = dino["feat_delta"].reshape(-1).astype(np.float32)

    parts = [feat_t, feat_delta]

    # Optionally include delta_z for hybrid mode
    if latent_dir is not None:
        lat = np.load(latent_dir / f"{sid}.npz")
        parts.append(lat["delta_z"].reshape(-1).astype(np.float32))

    # Proprioceptive features
    action = np.array(row.get("action", [0] * 7), dtype=np.float32)
    ee_delta = np.array(row.get("ee_delta", [0] * 6), dtype=np.float32)

    gripper_cmd = action[6] if len(action) > 6 else 0.0
    xyz_norm = float(np.linalg.norm(ee_delta[:3]))
    z_comp = ee_delta[2] if len(ee_delta) > 2 else 0.0
    rot_norm = float(np.linalg.norm(ee_delta[3:6])) if len(ee_delta) > 3 else 0.0
    action_xyz_norm = float(np.linalg.norm(action[:3]))

    gripper_is_closing = float(gripper_cmd < 0.3)
    gripper_is_opening = float(gripper_cmd > 0.7)
    low_movement = float(xyz_norm < 0.01)
    grasp_signal = gripper_is_closing * low_movement
    place_signal = gripper_is_opening * low_movement

    proprio = np.array([
        *action, *ee_delta,
        gripper_cmd, xyz_norm, z_comp, rot_norm, action_xyz_norm,
        gripper_is_closing, gripper_is_opening, low_movement,
        grasp_signal, place_signal,
    ], dtype=np.float32)

    parts.append(proprio)
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--dinov2_dir", type=Path, required=True)
    parser.add_argument("--latent_dir", type=Path, default=None, help="Optional: include delta_z for hybrid mode")
    parser.add_argument("--out_train", type=Path, required=True)
    parser.add_argument("--out_eval", type=Path, required=True)
    args = parser.parse_args()

    train_rows = [json.loads(l) for l in args.train_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in args.eval_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"Train: {len(train_rows)}, Eval: {len(eval_rows)}")

    x_train = np.stack([load_features(args.dinov2_dir, args.latent_dir, r) for r in train_rows])
    x_eval = np.stack([load_features(args.dinov2_dir, args.latent_dir, r) for r in eval_rows])
    print(f"Feature dim: {x_train.shape[1]}")

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_eval_s = scaler.transform(x_eval)

    by_id_train = {r["sample_id"]: {"sample_id": r["sample_id"], "gold_semantics": {}, "predicted_semantics": {}, "predicted_probabilities": {}} for r in train_rows}
    by_id_eval = {r["sample_id"]: {"sample_id": r["sample_id"], "gold_semantics": {}, "predicted_semantics": {}, "predicted_probabilities": {}} for r in eval_rows}

    for target in SEMANTIC_FIELDS:
        y_train = np.asarray([r["labels"][target] for r in train_rows])
        y_eval = np.asarray([r["labels"][target] for r in eval_rows])

        clf = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.03, max_depth=6,
            min_samples_leaf=5, l2_regularization=0.1,
            class_weight="balanced", random_state=42,
        )

        train_proba_cv = cross_val_predict(clf, x_train_s, y_train, cv=5, method="predict_proba", n_jobs=-1)
        clf.fit(x_train_s, y_train)
        classes = clf.classes_
        train_preds_cv = classes[np.argmax(train_proba_cv, axis=1)]

        eval_proba = clf.predict_proba(x_eval_s)
        eval_preds = clf.predict(x_eval_s)

        print(f"\n=== {target} (CV train) ===")
        print(classification_report(y_train, train_preds_cv, digits=4, zero_division=0))
        print(f"=== {target} (eval) ===")
        print(classification_report(y_eval, eval_preds, digits=4, zero_division=0))

        for i, row in enumerate(train_rows):
            sid = row["sample_id"]
            by_id_train[sid]["gold_semantics"][target] = row["labels"][target]
            by_id_train[sid]["predicted_semantics"][target] = train_preds_cv[i]
            by_id_train[sid]["predicted_probabilities"][target] = {cls: round(float(train_proba_cv[i, j]), 4) for j, cls in enumerate(classes)}

        for i, row in enumerate(eval_rows):
            sid = row["sample_id"]
            by_id_eval[sid]["gold_semantics"][target] = row["labels"][target]
            by_id_eval[sid]["predicted_semantics"][target] = eval_preds[i]
            by_id_eval[sid]["predicted_probabilities"][target] = {cls: round(float(eval_proba[i, j]), 4) for j, cls in enumerate(classes)}

    for path, by_id in [(args.out_train, by_id_train), (args.out_eval, by_id_eval)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for sid in sorted(by_id):
                f.write(json.dumps(by_id[sid]) + "\n")
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
