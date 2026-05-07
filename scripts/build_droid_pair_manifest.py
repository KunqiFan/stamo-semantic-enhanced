from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tfrecord.reader import tfrecord_loader


ROOT = Path(__file__).resolve().parents[1]


STAGE_LABELS = ["approach", "contact", "grasp", "lift", "move", "place"]


@dataclass
class EpisodeData:
    episode_id: str
    instruction: str
    action: np.ndarray
    cartesian_position: np.ndarray
    gripper_position: np.ndarray
    images: list[bytes]


def decode_text(value: object) -> str:
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", errors="ignore").strip()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def choose_instruction(record: dict) -> str:
    for key in [
        "steps/language_instruction",
        "steps/language_instruction_2",
        "steps/language_instruction_3",
    ]:
        if key not in record:
            continue
        for raw in record[key]:
            text = decode_text(raw)
            if text:
                return text
    return ""


def reshape_feature(record: dict, key: str, steps: int, width: int) -> np.ndarray:
    values = np.asarray(record[key], dtype=np.float32)
    if values.size != steps * width:
        raise ValueError(f"{key} has {values.size} values but expected {steps * width}")
    return values.reshape(steps, width)


def list_completed_shards(droid_dir: Path) -> list[Path]:
    candidates = sorted(droid_dir.glob("r2d2_faceblur-train.tfrecord-*-of-00031"))
    return [path for path in candidates if not path.name.endswith(".gstmp")]


def iter_episodes(droid_dir: Path, camera_key: str, max_episodes: int | None, require_language: bool) -> Iterable[EpisodeData]:
    count = 0
    image_field = f"steps/observation/{camera_key}"

    for shard_path in list_completed_shards(droid_dir):
        for record_index, record in enumerate(tfrecord_loader(str(shard_path), None)):
            instruction = choose_instruction(record)
            if require_language and not instruction:
                continue

            steps = len(record["steps/observation/gripper_position"])
            if steps < 2 or image_field not in record:
                continue

            try:
                action = reshape_feature(record, "steps/action", steps, 7)
                cartesian = reshape_feature(record, "steps/observation/cartesian_position", steps, 6)
            except ValueError:
                continue

            gripper = np.asarray(record["steps/observation/gripper_position"], dtype=np.float32)
            images = [bytes(raw) for raw in record[image_field]]
            if len(images) != steps:
                continue

            shard_tag = shard_path.name.replace(".", "_")
            episode_id = f"{shard_tag}_ep{record_index:03d}"
            yield EpisodeData(
                episode_id=episode_id,
                instruction=instruction or "short horizon manipulation",
                action=action,
                cartesian_position=cartesian,
                gripper_position=gripper,
                images=images,
            )
            count += 1
            if max_episodes is not None and count >= max_episodes:
                return


def classify_gripper_state(start_value: float, end_value: float) -> str:
    delta = end_value - start_value
    if end_value >= 0.65 and delta >= -0.05:
        return "open"
    if delta <= -0.08:
        return "closing"
    if end_value <= 0.2:
        return "closed"
    return "open" if end_value >= 0.45 else "closed"


def classify_contact_state(gripper_state: str, gripper_start: float, gripper_end: float) -> str:
    if gripper_state in {"closing", "closed"}:
        return "contact"
    if gripper_start > 0.75 and gripper_end > 0.75:
        return "no_contact"
    return "contact" if gripper_end < 0.35 else "no_contact"


def classify_target_relation_v1(cartesian: np.ndarray, start_idx: int, end_idx: int) -> str:
    """Original proxy: distance to episode end. Kept for comparison."""
    goal = cartesian[-1, :3]
    start_dist = float(np.linalg.norm(cartesian[start_idx, :3] - goal))
    end_dist = float(np.linalg.norm(cartesian[end_idx, :3] - goal))
    if end_dist < 0.035:
        return "reached"
    if end_dist < start_dist - 0.005:
        return "closer"
    if end_dist > start_dist + 0.005:
        return "farther"
    return "closer" if end_dist <= start_dist else "farther"


def classify_target_relation(
    cartesian: np.ndarray,
    gripper_position: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> str:
    """Improved proxy: combines local velocity trend + gripper activity + goal distance.

    Key changes over v1:
    - Local velocity trend: if robot is accelerating forward → closer
    - Gripper activity: significant grip change with low displacement → reached
    - Goal distance: used as tiebreaker with adaptive threshold
    """
    pos_start = cartesian[start_idx, :3]
    pos_end = cartesian[end_idx, :3]
    delta_pos = pos_end - pos_start
    move_mag = float(np.linalg.norm(delta_pos))

    # --- Signal 1: gripper activity at low displacement → reached ---
    grip_delta = abs(float(gripper_position[end_idx]) - float(gripper_position[start_idx]))
    if move_mag < 0.012 and grip_delta > 0.08:
        return "reached"

    # --- Signal 2: local velocity trend ---
    # Velocity before this pair
    lookback = max(0, start_idx - (end_idx - start_idx))
    vel_before = float(np.linalg.norm(pos_start - cartesian[lookback, :3]))
    vel_during = move_mag

    # Velocity after this pair (if available)
    lookahead = min(len(cartesian) - 1, end_idx + (end_idx - start_idx))
    vel_after = float(np.linalg.norm(cartesian[lookahead, :3] - pos_end))

    # Near-zero velocity in all windows → reached (settled at target)
    if vel_before < 0.008 and vel_during < 0.008 and vel_after < 0.008:
        return "reached"

    # --- Signal 3: goal distance as tiebreaker ---
    goal = cartesian[-1, :3]
    start_dist = float(np.linalg.norm(pos_start - goal))
    end_dist = float(np.linalg.norm(pos_end - goal))

    # Use relative threshold (5% of average distance) instead of fixed 0.005m
    avg_dist = (start_dist + end_dist) / 2.0 + 1e-6
    rel_threshold = max(0.005, avg_dist * 0.05)

    if end_dist < 0.035:
        return "reached"
    if end_dist < start_dist - rel_threshold:
        return "closer"
    if end_dist > start_dist + rel_threshold:
        return "farther"

    # Tie → use velocity direction consistency as final signal
    if vel_during > vel_before + 0.003:
        return "closer"   # accelerating → purposeful approach
    if vel_during < vel_before - 0.003:
        return "farther"  # decelerating → possibly retreating

    return "closer" if end_dist <= start_dist else "farther"


def classify_object_motion(cartesian: np.ndarray, gripper_state: str, start_idx: int, end_idx: int) -> str:
    delta_xyz = cartesian[end_idx, :3] - cartesian[start_idx, :3]
    move_norm = float(np.linalg.norm(delta_xyz))
    z_delta = float(delta_xyz[2])
    near_goal = float(np.linalg.norm(cartesian[end_idx, :3] - cartesian[-1, :3])) < 0.04

    if near_goal and gripper_state == "open":
        return "placed"
    if z_delta > 0.04 and move_norm > 0.05:
        return "lifted"
    if move_norm > 0.03:
        return "moved"
    return "still"


def derive_stage_label(contact_state: str, gripper_state: str, object_motion: str, target_relation: str) -> str:
    if object_motion == "placed" or (target_relation == "reached" and gripper_state == "open"):
        return "place"
    if object_motion == "lifted":
        return "lift"
    if gripper_state == "closing":
        return "grasp"
    if contact_state == "contact" and gripper_state == "open":
        return "contact"
    if object_motion == "moved":
        return "move"
    return "approach"


def make_process_text(task_text: str, stage_label: str) -> str:
    phase_hint = {
        "approach": "The robot is moving toward the object for the task.",
        "contact": "The robot is entering a brief interaction with the object.",
        "grasp": "The robot is forming a grasp during the task.",
        "lift": "The robot is lifting the object as part of the task.",
        "move": "The robot is carrying or repositioning the object.",
        "place": "The robot is completing placement near the task goal.",
    }[stage_label]
    return f"Task: {task_text}. {phase_hint}"


def save_image(image_bytes: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    output_path.write_bytes(image_bytes)


def make_sample(
    episode: EpisodeData,
    start_idx: int,
    end_idx: int,
    image_root: Path,
) -> dict:
    sample_id = f"{episode.episode_id}_{start_idx:04d}_{end_idx:04d}"
    image_t = image_root / episode.episode_id / f"{start_idx:04d}.jpg"
    image_tp = image_root / episode.episode_id / f"{end_idx:04d}.jpg"
    save_image(episode.images[start_idx], image_t)
    save_image(episode.images[end_idx], image_tp)

    gripper_state = classify_gripper_state(
        float(episode.gripper_position[start_idx]),
        float(episode.gripper_position[end_idx]),
    )
    contact_state = classify_contact_state(
        gripper_state,
        float(episode.gripper_position[start_idx]),
        float(episode.gripper_position[end_idx]),
    )
    target_relation = classify_target_relation(episode.cartesian_position, episode.gripper_position, start_idx, end_idx)
    object_motion = classify_object_motion(episode.cartesian_position, gripper_state, start_idx, end_idx)
    stage_label = derive_stage_label(contact_state, gripper_state, object_motion, target_relation)

    ee_delta = (
        episode.cartesian_position[end_idx, :6] - episode.cartesian_position[start_idx, :6]
    ).astype(np.float32)
    action = episode.action[start_idx:end_idx].mean(axis=0).astype(np.float32)

    return {
        "sample_id": sample_id,
        "task_name": episode.instruction,
        "image_t": str(image_t.resolve()),
        "image_tp": str(image_tp.resolve()),
        "trajectory_id": episode.episode_id,
        "start_step": start_idx,
        "end_step": end_idx,
        "caption": episode.instruction,
        "process_text": make_process_text(episode.instruction, stage_label),
        "action": action.tolist(),
        "ee_delta": ee_delta.tolist(),
        "labels": {
            "stage_label": stage_label,
            "contact_state": contact_state,
            "gripper_state": gripper_state,
            "object_motion": object_motion,
            "target_relation": target_relation,
        },
    }


def split_episodes(episodes: list[EpisodeData]) -> dict[str, list[EpisodeData]]:
    total = len(episodes)
    train_end = max(1, math.floor(total * 0.7))
    val_end = max(train_end + 1, math.floor(total * 0.85))
    return {
        "train": episodes[:train_end],
        "val": episodes[train_end:val_end],
        "test": episodes[val_end:],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def summarize(rows: list[dict]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {"stage_label": {}}
    for field in ["contact_state", "gripper_state", "object_motion", "target_relation"]:
        summary[field] = {}

    for row in rows:
        summary["stage_label"][row["labels"]["stage_label"]] = summary["stage_label"].get(row["labels"]["stage_label"], 0) + 1
        for field in ["contact_state", "gripper_state", "object_motion", "target_relation"]:
            value = row["labels"][field]
            summary[field][value] = summary[field].get(value, 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--droid_dir", type=Path, default=ROOT / "data" / "raw" / "droid_100" / "1.0.0")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "data" / "processed" / "droid_100_process_chain")
    parser.add_argument("--camera_key", type=str, default="wrist_image_left")
    parser.add_argument("--step_gap", type=int, default=4)
    parser.add_argument("--pair_stride", type=int, default=8)
    parser.add_argument("--max_episodes", type=int, default=24)
    parser.add_argument("--require_language", action="store_true")
    args = parser.parse_args()

    episodes = list(iter_episodes(args.droid_dir, args.camera_key, args.max_episodes, args.require_language))
    if len(episodes) < 3:
        raise RuntimeError("Not enough DROID episodes found to build train/val/test splits.")

    image_root = args.out_dir / "images"
    split_to_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for split_name, split_episodes_list in split_episodes(episodes).items():
        for episode in split_episodes_list:
            for start_idx in range(0, len(episode.images) - args.step_gap, args.pair_stride):
                end_idx = start_idx + args.step_gap
                split_to_rows[split_name].append(make_sample(episode, start_idx, end_idx, image_root))

    for split_name, rows in split_to_rows.items():
        write_jsonl(args.out_dir / f"{split_name}.jsonl", rows)
        print(f"{split_name}: wrote {len(rows)} samples")
        print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
