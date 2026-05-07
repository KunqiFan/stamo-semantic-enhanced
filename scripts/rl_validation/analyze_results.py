"""Analyze RL validation experiment results.

Reads log.json files from each run, generates:
1. Learning curves (success rate vs timesteps)
2. Sample efficiency table
3. Training overhead comparison
4. Statistical significance (Welch's t-test)

Usage:
    cd "stamo_pro - 副本 - 副本"
    py rl_validation/scripts/analyze_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "rl_validation" / "results"
PLOTS_DIR = ROOT / "rl_validation" / "plots"

TASKS = ["pickcube_v1", "stackcube_v1"]
GROUPS = {"diffonly": "Group A (diff-only)", "semantic": "Group B (diff+sem)"}
SEEDS = [42, 123, 456]


def load_logs(task: str, group: str) -> list[list[dict]]:
    """Load log.json for all seeds of a task/group combination."""
    runs = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"{task}_{group}_s{seed}" / "log.json"
        if path.exists():
            with open(path) as f:
                runs.append(json.load(f))
    return runs


def interpolate_metric(runs: list[list[dict]], metric: str,
                       steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a metric across seeds at common step values."""
    values = []
    for run in runs:
        run_steps = [e["step"] for e in run]
        run_vals = [e[metric] for e in run]
        interp = np.interp(steps, run_steps, run_vals)
        values.append(interp)
    values = np.array(values)
    return values.mean(axis=0), values.std(axis=0)


def find_threshold_step(runs: list[list[dict]], metric: str,
                        threshold: float) -> list[float]:
    """For each seed, find the first step where metric >= threshold."""
    results = []
    for run in runs:
        found = None
        for entry in run:
            if entry[metric] >= threshold:
                found = entry["step"]
                break
        results.append(found)
    return results


def welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test for unequal variances."""
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se < 1e-10:
        return 0.0, 1.0
    t_stat = (ma - mb) / se
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = df_num / max(df_den, 1e-10)
    from math import gamma as gamma_fn, pi, sqrt
    def t_cdf(t, v):
        x = v / (v + t * t)
        from functools import reduce
        def beta_inc(a, b, x):
            if x <= 0:
                return 0.0
            if x >= 1:
                return 1.0
            n_terms = 200
            result = 0.0
            for k in range(n_terms):
                coeff = 1.0
                for j in range(k):
                    coeff *= (j - b + 1) / (j + 1)
                result += coeff * x ** k / (a + k)
            return x ** a * result
        try:
            bt = gamma_fn(v / 2 + 0.5) / (sqrt(pi * v) * gamma_fn(v / 2))
            p = 0.5 + 0.5 * np.sign(t) * (1 - beta_inc(v / 2, 0.5, x))
        except Exception:
            p = 0.5
        return p
    p_value = 2 * (1 - t_cdf(abs(t_stat), df))
    p_value = max(0, min(1, p_value))
    return float(t_stat), float(p_value)


def plot_learning_curves(task: str):
    """Generate learning curve plot for one task."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_steps = set()
    for group in GROUPS:
        for run in load_logs(task, group):
            for entry in run:
                all_steps.add(entry["step"])
    if not all_steps:
        print(f"  No data for {task}")
        return
    steps = np.array(sorted(all_steps))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = {"diffonly": "#2196F3", "semantic": "#FF5722"}

    for group, label in GROUPS.items():
        runs = load_logs(task, group)
        if not runs:
            continue
        mean, std = interpolate_metric(runs, "success_rate", steps)
        ax.plot(steps, mean, label=label, color=colors[group], linewidth=2)
        ax.fill_between(steps, mean - std, mean + std,
                        alpha=0.2, color=colors[group])

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Learning Curves: {task}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / f"{task}_learning_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {task}_learning_curves.png")


def print_summary():
    """Print summary tables to console."""
    print("\n" + "=" * 70)
    print("RL VALIDATION EXPERIMENT RESULTS")
    print("=" * 70)

    for task in TASKS:
        print(f"\n--- {task} ---\n")

        # Final success rates
        final_rates = {}
        for group in GROUPS:
            runs = load_logs(task, group)
            rates = []
            for run in runs:
                if run:
                    rates.append(run[-1]["success_rate"])
            final_rates[group] = rates
            mean = np.mean(rates) if rates else 0
            std = np.std(rates) if rates else 0
            print(f"  {GROUPS[group]:30s}: {mean:.1%} +/- {std:.1%}  (n={len(rates)})")

        # Statistical test
        a = final_rates.get("diffonly", [])
        b = final_rates.get("semantic", [])
        if len(a) >= 2 and len(b) >= 2:
            t_stat, p_val = welch_t_test(a, b)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"  Welch's t-test: t={t_stat:.3f}, p={p_val:.4f} {sig}")

        # Sample efficiency
        print(f"\n  Sample efficiency (steps to reach threshold):")
        for threshold in [0.5, 0.8]:
            print(f"    Threshold = {threshold:.0%}:")
            for group in GROUPS:
                runs = load_logs(task, group)
                steps_list = find_threshold_step(runs, "success_rate", threshold)
                reached = [s for s in steps_list if s is not None]
                if reached:
                    print(f"      {GROUPS[group]:30s}: {np.mean(reached):.0f} "
                          f"+/- {np.std(reached):.0f} steps  "
                          f"({len(reached)}/{len(steps_list)} seeds reached)")
                else:
                    print(f"      {GROUPS[group]:30s}: not reached")

        # Training overhead
        print(f"\n  Training overhead:")
        for group in GROUPS:
            runs = load_logs(task, group)
            wall_clocks = []
            enc_times = []
            for run in runs:
                if run:
                    wall_clocks.append(run[-1]["wall_clock"])
                    enc_times.extend(e.get("encoder_time_per_rollout", 0) for e in run)
            if wall_clocks:
                print(f"    {GROUPS[group]:30s}: "
                      f"wall_clock={np.mean(wall_clocks)/3600:.1f}h, "
                      f"enc_time/rollout={np.mean(enc_times):.1f}s")


def main():
    print_summary()
    for task in TASKS:
        plot_learning_curves(task)
    print(f"\nPlots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
