"""Quick eval for Position C (dino) latents."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stamo_bridge.semantics.interface import SEMANTIC_FIELDS

latent_dir = ROOT / "data" / "interim" / "droid_process_chain" / "latents_pos_dino"
train_manifest = ROOT / "data" / "processed" / "droid_process_chain" / "train.jsonl"
eval_manifest = ROOT / "data" / "processed" / "droid_process_chain" / "test.jsonl"

train_rows = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
eval_rows = [json.loads(l) for l in eval_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

def load_visual(row):
    data = np.load(latent_dir / f"{row['sample_id']}.npz")
    dz = data["delta_z"].reshape(-1).astype(np.float32)
    dp = data["delta_pooled"].reshape(-1).astype(np.float32)
    return np.concatenate([dz, dp])

x_train = np.stack([load_visual(r) for r in train_rows])
x_eval = np.stack([load_visual(r) for r in eval_rows])
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_eval_s = scaler.transform(x_eval)

for clf_name, clf_cls in [
    ("HistGBT", lambda: HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.03, max_depth=6,
        min_samples_leaf=5, l2_regularization=0.1,
        class_weight="balanced", random_state=42)),
    ("LogReg", lambda: LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced",
        random_state=42, solver="lbfgs")),
]:
    print(f"\n=== Position dino -- {clf_name} (visual-only 2560D) ===")
    accs = {}
    for target in SEMANTIC_FIELDS:
        y_train = np.asarray([r["labels"][target] for r in train_rows])
        y_eval = np.asarray([r["labels"][target] for r in eval_rows])
        clf = clf_cls()
        clf.fit(x_train_s, y_train)
        acc = (clf.predict(x_eval_s) == y_eval).mean()
        accs[target] = float(acc)
        print(f"  {target}: {acc:.4f}")
    accs["mean"] = float(np.mean(list(accs.values())))
    print(f"  mean: {accs['mean']:.4f}")
