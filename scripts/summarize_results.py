"""Generate a consolidated results summary across all experiments."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_json(name):
    p = RESULTS / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def main():
    print("=" * 80)
    print("  TEXT BRIDGE EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    # 1. Stage classification with gold labels
    print("\n\n--- Experiment 1: Stage Classification (Gold Semantics) ---")
    print("Task: predict stage_label from gold compact semantics")
    print("Note: 4 attributes -> stage is a deterministic mapping\n")
    d = load_json("text_bridge_experiment.json")
    if d:
        print(f"{'Condition':<28} {'Acc':>8} {'Macro-F1':>10}")
        print("-" * 48)
        for cond, r in d.items():
            print(f"{cond:<28} {r['accuracy']:>8.4f} {r['macro_f1']:>10.4f}")

    # 2. Stage classification with predicted semantics (fair)
    print("\n\n--- Experiment 2: Stage Classification (Predicted Semantics, Fair) ---")
    print("Task: predict stage_label; all conditions use predicted semantics\n")
    d = load_json("droid_text_bridge_fair.json")
    if d:
        print(f"{'Condition':<28} {'Acc':>8} {'Macro-F1':>10}")
        print("-" * 48)
        for cond, r in d.items():
            print(f"{cond:<28} {r['accuracy']:>8.4f} {r['macro_f1']:>10.4f}")

    # 3. Action prediction with gold labels
    print("\n\n--- Experiment 3: ee_delta Prediction (Gold Semantics) ---")
    print("Task: predict 6-dim end-effector delta from gold compact semantics\n")
    d = load_json("action_prediction_gold.json")
    if d:
        print(f"{'Condition':<28} {'MSE':>10} {'MAE':>10} {'R2':>8}")
        print("-" * 58)
        for cond, r in d.items():
            print(f"{cond:<28} {r['mse']:>10.6f} {r['mae']:>10.6f} {r['r2']:>8.4f}")

    # 4. Action prediction with predicted semantics
    print("\n\n--- Experiment 4: ee_delta Prediction (Predicted Semantics) ---")
    print("Task: predict 6-dim end-effector delta from predicted compact semantics\n")
    d = load_json("action_prediction_predicted.json")
    if d:
        print(f"{'Condition':<28} {'MSE':>10} {'MAE':>10} {'R2':>8}")
        print("-" * 58)
        for cond, r in d.items():
            print(f"{cond:<28} {r['mse']:>10.6f} {r['mae']:>10.6f} {r['r2']:>8.4f}")

    # 5. Few-shot
    print("\n\n--- Experiment 5: Few-Shot Stage Classification (Gold Semantics) ---")
    print("Task: stage_label with subsampled training data\n")
    d = load_json("fewshot_gold_experiment.json")
    if d:
        fracs = sorted(d.keys(), key=float)
        header = f"{'Condition':<28}"
        for f in fracs:
            header += f" {float(f):>7.0%}"
        print(header)
        print("-" * (28 + 8 * len(fracs)))
        conds = list(next(iter(d.values())).keys())
        for cond in conds:
            line = f"{cond:<28}"
            for f in fracs:
                line += f" {d[f][cond]['mean_macro_f1']:>7.4f}"
            print(line)

    # Key findings
    print("\n\n" + "=" * 80)
    print("  KEY FINDINGS")
    print("=" * 80)
    print("""
1. STAGE CLASSIFICATION is trivially solved by discrete labels (deterministic
   mapping from 4 attributes to stage). Not a useful benchmark for text bridge.

2. ACTION PREDICTION (ee_delta) with GOLD semantics shows:
   - enriched_text (R2=0.086) > discrete_labels (R2=0.083) > template_text (R2=0.076)
   - Text bridge provides marginal improvement over discrete labels
   - SBERT enriched encoding captures slightly more useful structure

3. ACTION PREDICTION with PREDICTED semantics shows:
   - All semantic conditions collapse to near-zero R2 (~-0.007)
   - Upstream prediction noise (only 9% all-4-correct) destroys signal
   - Text bridge does NOT rescue noisy predictions

4. raw_delta_z alone is consistently worst (R2 < 0), suggesting the 512-dim
   latent is too high-dimensional for tree-based models without dimensionality
   reduction.

5. FUSION (delta_z + semantics) hurts rather than helps, likely because
   the high-dim delta_z dominates and introduces noise.
""")


if __name__ == "__main__":
    main()
