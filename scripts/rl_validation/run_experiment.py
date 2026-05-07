"""Orchestrate the full RL validation experiment.

Runs all combinations of {task} x {group} x {seed} sequentially.
Each run: PPO training with StaMo encoder, logging to results/.

Usage:
    cd "stamo_pro - 副本 - 副本"
    py rl_validation/scripts/run_experiment.py
    py rl_validation/scripts/run_experiment.py --task PickCube-v1 --seeds 42
    py rl_validation/scripts/run_experiment.py --skip_data --skip_stamo
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = ROOT / "rl_validation"
STAMO_ROOT = ROOT / "StaMo"

TASKS = ["PickCube-v1", "StackCube-v1"]
GROUPS = ["A", "B"]
SEEDS = [42, 123, 456]

TASK_TIMESTEPS = {
    "PickCube-v1": 500_000,
    "StackCube-v1": 1_000_000,
}

TASK_SLUG = {
    "PickCube-v1": "pickcube_v1",
    "StackCube-v1": "stackcube_v1",
}


def run_cmd(cmd: list[str], desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n  [{status}] {desc} ({elapsed:.0f}s)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=TASKS + ["all"])
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated seeds")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Episodes for data collection")
    parser.add_argument("--stamo_iters", type=int, default=5000)
    parser.add_argument("--skip_data", action="store_true")
    parser.add_argument("--skip_stamo", action="store_true")
    parser.add_argument("--skip_rl", action="store_true")
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    seeds = [int(s) for s in args.seeds.split(",")]
    py = sys.executable

    # Phase 1: Data collection
    if not args.skip_data:
        for task in tasks:
            ok = run_cmd(
                [py, str(RL_ROOT / "scripts" / "collect_maniskill_data.py"),
                 "--task", task, "--episodes", str(args.episodes)],
                f"Phase 1: Collect data for {task}")
            if not ok:
                print(f"Data collection failed for {task}, aborting.")
                return

    # Phase 2: StaMo training
    if not args.skip_stamo:
        for task in tasks:
            for group in GROUPS:
                ok = run_cmd(
                    [py, str(RL_ROOT / "scripts" / "train_stamo_maniskill.py"),
                     "--group", group, "--task", task,
                     "--num_iters", str(args.stamo_iters)],
                    f"Phase 2: Train StaMo Group {group} on {task}")
                if not ok:
                    print(f"StaMo training failed for {task} Group {group}, aborting.")
                    return

    # Phase 3: RL training
    if not args.skip_rl:
        config_path = str(RL_ROOT / "configs" / "stamo_maniskill.yaml")
        for task in tasks:
            slug = TASK_SLUG[task]
            timesteps = TASK_TIMESTEPS[task]
            for group in GROUPS:
                group_name = "diffonly" if group == "A" else "semantic"
                ckpt_dir = str(STAMO_ROOT / "ckpts" /
                               f"maniskill_{slug}_{group_name}" /
                               str(args.stamo_iters))
                if not Path(ckpt_dir).exists():
                    print(f"Checkpoint not found: {ckpt_dir}, skipping.")
                    continue
                for seed in seeds:
                    run_name = f"{slug}_{group_name}_s{seed}"
                    ok = run_cmd(
                        [py, str(RL_ROOT / "scripts" / "ppo_stamo.py"),
                         "--task", task,
                         "--checkpoint", ckpt_dir,
                         "--config", config_path,
                         "--total_timesteps", str(timesteps),
                         "--seed", str(seed),
                         "--run_name", run_name],
                        f"Phase 3: PPO {task} Group {group} seed={seed}")
                    if not ok:
                        print(f"PPO failed for {run_name}, continuing...")

    print(f"\n{'='*60}")
    print("Experiment complete. Run analyze_results.py to generate plots.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
