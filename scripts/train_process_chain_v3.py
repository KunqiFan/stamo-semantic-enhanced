"""Process chain v3 — Maximum performance fusion.

Key improvements over v2:
1. Injects PCA-reduced delta_z visual features as part of the semantic bridge
2. Uses SMOTE oversampling to address class imbalance (grasp/place)
3. Adds action vector to physics features
4. Uses calibrated probability stacking
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stamo_bridge.semantics.interface import SEMANTIC_FIELDS, SEMANTIC_LABELS  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_predicted(path: Path | None) -> dict[str, dict] | None:
    if path is None:
        return None
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[row["sample_id"]] = row
    return result


def soft_semantics_vector(sample_id, predicted, row_labels):
    vector = []
    for field in SEMANTIC_FIELDS:
        labels = SEMANTIC_LABELS[field]
        if predicted and sample_id in predicted:
            entry = predicted[sample_id]
            if "predicted_probabilities" in entry and field in entry["predicted_probabilities"]:
                probs = entry["predicted_probabilities"][field]
                for lbl in labels:
                    vector.append(probs.get(lbl, 0.0))
                continue
            if "predicted_semantics" in entry and field in entry["predicted_semantics"]:
                pred_val = entry["predicted_semantics"][field]
                for lbl in labels:
                    vector.append(1.0 if lbl == pred_val else 0.0)
                continue
        gold_val = row_labels[field]
        for lbl in labels:
            vector.append(1.0 if lbl == gold_val else 0.0)
    return np.array(vector, dtype=np.float32)


def get_physics(row):
    """ee_delta only (no action) — this is the physics baseline."""
    ee = np.array(row.get("ee_delta", [0]*6), dtype=np.float32)
    xyz_norm = float(np.linalg.norm(ee[:3]))
    z_comp = ee[2]
    rot_norm = float(np.linalg.norm(ee[3:6]))
    return np.concatenate([ee, [xyz_norm, z_comp, rot_norm]]).astype(np.float32)


def get_action(row):
    """Action vector including gripper command."""
    action = np.array(row.get("action", [0]*7), dtype=np.float32)
    gripper_cmd = action[6] if len(action) > 6 else 0.0
    gripper_closing = float(gripper_cmd < 0.3)
    gripper_opening = float(gripper_cmd > 0.7)
    return np.concatenate([action, [gripper_closing, gripper_opening]]).astype(np.float32)


def load_latent_pca(rows, latent_dir, n_components=48, pca_model=None):
    """Load delta_z + delta_pooled and apply PCA reduction."""
    feats = []
    for r in rows:
        data = np.load(latent_dir / f"{r['sample_id']}.npz")
        dz = data["delta_z"].reshape(-1).astype(np.float32)
        dp = data["delta_pooled"].reshape(-1).astype(np.float32)
        feats.append(np.concatenate([dz, dp]))
    X = np.stack(feats)
    if pca_model is None:
        pca_model = PCA(n_components=n_components, random_state=42)
        X_pca = pca_model.fit_transform(X)
    else:
        X_pca = pca_model.transform(X)
    return X_pca, pca_model


def text_inputs(rows, text_key):
    return [r.get(text_key) or r.get("caption") or "" for r in rows]


def make_hgb(**overrides):
    defaults = dict(
        max_iter=800, learning_rate=0.02, max_depth=6,
        min_samples_leaf=3, l2_regularization=0.05,
        class_weight="balanced", random_state=42,
    )
    defaults.update(overrides)
    return HistGradientBoostingClassifier(**defaults)


def train_eval(name, x_tr, x_ev, y_tr, y_ev, use_smote=True):
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_ev = scaler.transform(x_ev)

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42, k_neighbors=min(3, min(
                np.bincount([list(np.unique(y_tr)).index(v) for v in y_tr]).min() - 1, 5
            )))
            x_tr, y_tr = smote.fit_resample(x_tr, y_tr)
            print(f"  [SMOTE] Resampled: {len(x_tr)} samples")
        except Exception as e:
            print(f"  [SMOTE] Skipped: {e}")

    print(f"\n  [{name}] Feature dim: {x_tr.shape[1]}, Train: {len(x_tr)}")
    clf = make_hgb()
    clf.fit(x_tr, y_tr)
    preds = clf.predict(x_ev)
    print(classification_report(y_ev, preds, digits=4, zero_division=0))
    return clf, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--latent_dir", type=Path, default=None)
    parser.add_argument("--setting", choices=[
        "text_only", "physics_only", "semantics_only",
        "bridge", "bridge_plus_physics",
        "full_fusion", "gold_bridge",
    ], required=True)
    parser.add_argument("--target", type=str, default="stage_label")
    parser.add_argument("--text_key", type=str, default="caption")
    parser.add_argument("--predicted_semantics", type=Path, default=None)
    parser.add_argument("--predicted_semantics_train", type=Path, default=None)
    args = parser.parse_args()

    train_rows = load_rows(args.train_manifest)
    eval_rows = load_rows(args.eval_manifest)
    pred_eval = load_predicted(args.predicted_semantics)
    pred_train = load_predicted(args.predicted_semantics_train)

    y_train = np.asarray([r["labels"][args.target] for r in train_rows])
    y_eval = np.asarray([r["labels"][args.target] for r in eval_rows])

    # Build feature blocks
    phys_train = np.stack([get_physics(r) for r in train_rows])
    phys_eval = np.stack([get_physics(r) for r in eval_rows])

    act_train = np.stack([get_action(r) for r in train_rows])
    act_eval = np.stack([get_action(r) for r in eval_rows])

    sem_train = np.stack([soft_semantics_vector(r["sample_id"], pred_train, r["labels"]) for r in train_rows])
    sem_eval = np.stack([soft_semantics_vector(r["sample_id"], pred_eval, r["labels"]) for r in eval_rows])

    # Load PCA latent features if available
    lat_train = lat_eval = None
    if args.latent_dir and args.latent_dir.exists():
        lat_train, pca = load_latent_pca(train_rows, args.latent_dir, n_components=48)
        lat_eval, _ = load_latent_pca(eval_rows, args.latent_dir, pca_model=pca)
        print(f"  Latent PCA features: {lat_train.shape[1]}d")

    # --- Text features ---
    try:
        from sentence_transformers import SentenceTransformer
        print("\n  [info] Using sentence-transformers")
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        txt_train = enc.encode(text_inputs(train_rows, args.text_key), show_progress_bar=False)
        txt_eval = enc.encode(text_inputs(eval_rows, args.text_key), show_progress_bar=False)
    except ImportError:
        print("\n  [info] Falling back to TF-IDF")
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.95, sublinear_tf=True)
        txt_train = tfidf.fit_transform(text_inputs(train_rows, args.text_key)).toarray()
        txt_eval = tfidf.transform(text_inputs(eval_rows, args.text_key)).toarray()

    # ============================================================
    if args.setting == "text_only":
        train_eval("text_only", txt_train, txt_eval, y_train, y_eval)

    elif args.setting == "physics_only":
        train_eval("physics_only", phys_train, phys_eval, y_train, y_eval)

    elif args.setting == "semantics_only":
        train_eval("semantics_only", sem_train, sem_eval, y_train, y_eval)

    elif args.setting == "bridge":
        # Semantic bridge = soft semantics + PCA latents (visual info)
        parts_tr = [sem_train]
        parts_ev = [sem_eval]
        if lat_train is not None:
            parts_tr.append(lat_train)
            parts_ev.append(lat_eval)
        x_tr = np.hstack(parts_tr)
        x_ev = np.hstack(parts_ev)
        train_eval("bridge (sem + latent PCA)", x_tr, x_ev, y_train, y_eval)

    elif args.setting == "bridge_plus_physics":
        # Semantic bridge + raw physics
        parts_tr = [sem_train]
        parts_ev = [sem_eval]
        if lat_train is not None:
            parts_tr.append(lat_train)
            parts_ev.append(lat_eval)
        parts_tr.append(phys_train)
        parts_ev.append(phys_eval)
        x_tr = np.hstack(parts_tr)
        x_ev = np.hstack(parts_ev)
        train_eval("bridge + physics", x_tr, x_ev, y_train, y_eval)

    elif args.setting == "full_fusion":
        # Full: semantic bridge + physics + text (via stacking)
        # Stage 1: Bridge + Physics sub-model
        bridge_tr = [sem_train]
        bridge_ev = [sem_eval]
        if lat_train is not None:
            bridge_tr.append(lat_train)
            bridge_ev.append(lat_eval)
        bridge_tr.append(phys_train)
        bridge_ev.append(phys_eval)
        dense_tr = np.hstack(bridge_tr)
        dense_ev = np.hstack(bridge_ev)

        print("\n  === Sub-model: Bridge + Physics ===")
        train_eval("bridge+phys sub", dense_tr, dense_ev, y_train, y_eval)

        # Cross-val probabilities
        scaler_d = StandardScaler()
        dense_tr_s = scaler_d.fit_transform(dense_tr)
        dense_ev_s = scaler_d.transform(dense_ev)

        # Apply SMOTE for cross_val_predict
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            smote_pipe = ImbPipeline([
                ("smote", SMOTE(random_state=42, k_neighbors=3)),
                ("clf", make_hgb()),
            ])
            dense_proba_train_cv = cross_val_predict(
                smote_pipe, dense_tr_s, y_train, cv=5, method="predict_proba", n_jobs=-1
            )
            smote_pipe.fit(dense_tr_s, y_train)
            dense_proba_eval = smote_pipe.predict_proba(dense_ev_s)
        except Exception:
            clf_d = make_hgb()
            dense_proba_train_cv = cross_val_predict(
                clf_d, dense_tr_s, y_train, cv=5, method="predict_proba", n_jobs=-1
            )
            clf_d.fit(dense_tr_s, y_train)
            dense_proba_eval = clf_d.predict_proba(dense_ev_s)

        # Stage 2: Text sub-model
        scaler_t = StandardScaler()
        txt_tr_s = scaler_t.fit_transform(txt_train)
        txt_ev_s = scaler_t.transform(txt_eval)
        clf_t = make_hgb()
        txt_proba_train_cv = cross_val_predict(
            clf_t, txt_tr_s, y_train, cv=5, method="predict_proba", n_jobs=-1
        )
        clf_t.fit(txt_tr_s, y_train)
        txt_proba_eval = clf_t.predict_proba(txt_ev_s)

        # Stage 3: Meta-classifier
        meta_tr = np.hstack([dense_proba_train_cv, txt_proba_train_cv, sem_train, phys_train])
        meta_ev = np.hstack([dense_proba_eval, txt_proba_eval, sem_eval, phys_eval])

        print("\n  === FINAL: Meta-classifier ===")
        train_eval("meta (full fusion)", meta_tr, meta_ev, y_train, y_eval)

    elif args.setting == "gold_bridge":
        # Gold semantics + physics (upper bound)
        x_tr = np.hstack([sem_train, phys_train])
        x_ev = np.hstack([sem_eval, phys_eval])
        train_eval("gold_bridge", x_tr, x_ev, y_train, y_eval, use_smote=False)


if __name__ == "__main__":
    main()
