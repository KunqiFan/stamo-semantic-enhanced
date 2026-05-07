"""Collect expert demonstrations from ManiSkill3 and build StaMo-compatible manifests.

Replays pre-recorded demo trajectories (downloaded via
``py -m mani_skill.utils.download_demo``), renders 224x224 RGB frames,
constructs semantic labels from ground-truth state, and writes JSONL
manifests for StaMo's PairImageData.

Usage:
    cd "stamo_pro - 副本 - 副本"
    py rl_validation/scripts/collect_maniskill_data.py --task PickCube-v1 --episodes 200
    py rl_validation/scripts/collect_maniskill_data.py --task StackCube-v1 --episodes 200
    py rl_validation/scripts/collect_maniskill_data.py --task all --episodes 200
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
from PIL import Image

import mani_skill.envs  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "rl_validation" / "data"

STEP_GAP = 4
PAIR_STRIDE = 2

DEMO_DIR = Path.home() / ".maniskill" / "demos"

TASKS = ["PickCube-v1", "StackCube-v1"]


# ── Semantic classifiers ────────────────────────────────────────────────────

def classify_contact(is_grasped: bool, tcp_to_obj_dist: float) -> str:
    if is_grasped or tcp_to_obj_dist < 0.02:
        return "contact"
    return "no_contact"


def classify_gripper(qpos: np.ndarray) -> str:
    finger_sum = float(qpos[-2:].sum())
    if finger_sum > 0.06:
        return "open"
    if finger_sum > 0.02:
        return "closing"
    return "closed"


def classify_motion(obj_pos_t: np.ndarray, obj_pos_tp: np.ndarray) -> str:
    delta = obj_pos_tp - obj_pos_t
    move_norm = float(np.linalg.norm(delta))
    z_delta = float(delta[2])
    if z_delta > 0.005 and move_norm > 0.008:
        return "lifted"
    if z_delta < -0.005 and move_norm > 0.008:
        return "placed"
    if move_norm > 0.005:
        return "moved"
    return "still"


def classify_relation_pick(obj_to_goal_t: np.ndarray,
                           obj_to_goal_tp: np.ndarray) -> str:
    dist_t = float(np.linalg.norm(obj_to_goal_t))
    dist_tp = float(np.linalg.norm(obj_to_goal_tp))
    if dist_tp < 0.03:
        return "reached"
    if dist_tp < dist_t - 0.003:
        return "closer"
    return "farther"


def classify_relation_stack(a_to_b_t: np.ndarray, a_to_b_tp: np.ndarray,
                            cube_half: float = 0.02) -> str:
    goal = np.array([0.0, 0.0, cube_half * 2])
    dist_t = float(np.linalg.norm(a_to_b_t + goal))
    dist_tp = float(np.linalg.norm(a_to_b_tp + goal))
    if dist_tp < 0.03:
        return "reached"
    if dist_tp < dist_t - 0.003:
        return "closer"
    return "farther"


# ── Frame capture helper ───────────────────────────────────────────────────

def get_render_camera(env):
    """Return the SAPIEN RenderCameraComponent for the base_camera sensor."""
    scene = env.unwrapped.scene
    return scene.sensors["base_camera"].camera._render_cameras[0]


def capture_frame(env, render_cam, size: int = 224) -> np.ndarray:
    """Render one RGB frame and resize to (size, size, 3) uint8."""
    env.unwrapped.scene.update_render()
    render_cam.take_picture()
    rgba = render_cam.get_picture("Color")
    rgb = (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = np.array(Image.fromarray(rgb).resize((size, size), Image.LANCZOS))
    return rgb


# ── State extraction ───────────────────────────────────────────────────────

def extract_state_pick(env) -> dict:
    uw = env.unwrapped
    return {
        "qpos": uw.agent.robot.get_qpos()[0].cpu().numpy().copy(),
        "is_grasped": bool(uw.agent.is_grasping(uw.cube)[0].item()),
        "tcp_pos": uw.agent.tcp.pose.p[0].cpu().numpy().copy(),
        "obj_pos": uw.cube.pose.p[0].cpu().numpy().copy(),
        "goal_pos": uw.goal_site.pose.p[0].cpu().numpy().copy(),
    }


def extract_state_stack(env) -> dict:
    uw = env.unwrapped
    return {
        "qpos": uw.agent.robot.get_qpos()[0].cpu().numpy().copy(),
        "is_grasped": bool(uw.agent.is_grasping(uw.cubeA)[0].item()),
        "tcp_pos": uw.agent.tcp.pose.p[0].cpu().numpy().copy(),
        "cubeA_pos": uw.cubeA.pose.p[0].cpu().numpy().copy(),
        "cubeB_pos": uw.cubeB.pose.p[0].cpu().numpy().copy(),
    }


# ── Label construction ─────────────────────────────────────────────────────

def make_labels_pick(st: dict, stp: dict) -> dict:
    tcp_to_obj = float(np.linalg.norm(st["obj_pos"] - st["tcp_pos"]))
    return {
        "contact_state": classify_contact(st["is_grasped"], tcp_to_obj),
        "gripper_state": classify_gripper(st["qpos"]),
        "object_motion": classify_motion(st["obj_pos"], stp["obj_pos"]),
        "target_relation": classify_relation_pick(
            st["goal_pos"] - st["obj_pos"], stp["goal_pos"] - stp["obj_pos"]),
    }


def make_labels_stack(st: dict, stp: dict) -> dict:
    tcp_to_obj = float(np.linalg.norm(st["cubeA_pos"] - st["tcp_pos"]))
    return {
        "contact_state": classify_contact(st["is_grasped"], tcp_to_obj),
        "gripper_state": classify_gripper(st["qpos"]),
        "object_motion": classify_motion(st["cubeA_pos"], stp["cubeA_pos"]),
        "target_relation": classify_relation_stack(
            st["cubeB_pos"] - st["cubeA_pos"],
            stp["cubeB_pos"] - stp["cubeA_pos"]),
    }


# ── Pair generation ─────────────────────────────────────────────────────────

def generate_pairs(frames: list[np.ndarray], states: list[dict],
                   task_id: str, ep_id: str, img_dir: Path,
                   is_stack: bool) -> list[dict]:
    """Save frames and generate image-pair samples from one episode."""
    make_labels = make_labels_stack if is_stack else make_labels_pick
    ep_img_dir = img_dir / ep_id
    ep_img_dir.mkdir(parents=True, exist_ok=True)

    n = len(frames)
    img_paths: list[str] = []
    for i, frame in enumerate(frames):
        p = ep_img_dir / f"{i:04d}.jpg"
        Image.fromarray(frame).save(str(p), quality=95)
        img_paths.append(str(p))

    pairs = []
    for start in range(0, n - STEP_GAP, PAIR_STRIDE):
        end = start + STEP_GAP
        if end >= n:
            break
        labels = make_labels(states[start], states[end])
        pairs.append({
            "sample_id": f"{ep_id}_{start:04d}_{end:04d}",
            "task_name": task_id,
            "image_t": img_paths[start],
            "image_tp": img_paths[end],
            "start_step": start,
            "end_step": end,
            "labels": labels,
        })
    return pairs


# ── Main collection loop ──────────────────────────────────────────────────

def collect_task(task_id: str, num_episodes: int):
    demo_h5_path = DEMO_DIR / task_id / "motionplanning" / "trajectory.h5"
    demo_json_path = DEMO_DIR / task_id / "motionplanning" / "trajectory.json"
    if not demo_h5_path.exists():
        raise FileNotFoundError(
            f"Demo not found: {demo_h5_path}\n"
            f"Download with: py -m mani_skill.utils.download_demo {task_id}")

    h5 = h5py.File(str(demo_h5_path), "r")
    meta = json.load(open(str(demo_json_path), encoding="utf-8"))
    episodes = meta["episodes"]

    is_stack = "Stack" in task_id
    extract_state = extract_state_stack if is_stack else extract_state_pick

    task_dir = DATA_DIR / task_id.lower().replace("-", "_")
    img_dir = task_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(task_id, obs_mode="state_dict", control_mode="pd_joint_pos")
    render_cam = get_render_camera(env)

    all_pairs: list[dict] = []
    use_episodes = min(num_episodes, len(episodes))

    for idx in range(use_episodes):
        ep = episodes[idx]
        traj_key = f"traj_{ep['episode_id']}"
        if traj_key not in h5:
            continue
        actions = h5[traj_key]["actions"][:]

        env.reset(seed=ep["episode_seed"])

        frames: list[np.ndarray] = []
        states: list[dict] = []

        frames.append(capture_frame(env, render_cam))
        states.append(extract_state(env))

        for action in actions:
            env.step(action)
            frames.append(capture_frame(env, render_cam))
            states.append(extract_state(env))

        ep_id = f"{task_id.lower().replace('-','_')}_ep{idx:04d}"
        pairs = generate_pairs(frames, states, task_id, ep_id, img_dir, is_stack)
        all_pairs.extend(pairs)

        done = idx + 1
        if done % 10 == 0 or done == use_episodes:
            print(f"  [{task_id}] {done}/{use_episodes} episodes, "
                  f"{len(all_pairs)} pairs so far")

    env.close()
    h5.close()

    random.shuffle(all_pairs)
    split = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split]
    test_pairs = all_pairs[split:]

    for path, data in [(task_dir / "train.jsonl", train_pairs),
                       (task_dir / "test.jsonl", test_pairs)]:
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n  [{task_id}] Done: {len(train_pairs)} train, {len(test_pairs)} test")
    print_label_stats(all_pairs)
    return all_pairs


def print_label_stats(pairs: list[dict]):
    for field in ["contact_state", "gripper_state", "object_motion", "target_relation"]:
        counts = Counter(p["labels"][field] for p in pairs)
        total = sum(counts.values())
        dist = ", ".join(f"{k}: {v} ({v/total:.0%})" for k, v in sorted(counts.items()))
        print(f"    {field}: {dist}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        choices=["PickCube-v1", "StackCube-v1", "all"])
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    for task_id in tasks:
        print(f"\n{'='*60}")
        print(f"Collecting {args.episodes} episodes for {task_id}")
        print(f"{'='*60}")
        collect_task(task_id, args.episodes)

    print("\nAll done.")


if __name__ == "__main__":
    main()
