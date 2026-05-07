"""Text bridge: convert compact process semantics into natural language descriptions.

Two levels:
- template_text: deterministic rule-based generation (same info as discrete labels)
- enriched_text: richer descriptions with physical/commonsense context (for LLM expansion)
"""
from __future__ import annotations

from dataclasses import dataclass

STAGE_TEMPLATES = {
    "approach": "The robot arm is approaching the target object, moving closer without making contact.",
    "contact": "The robot arm has just made contact with the object surface.",
    "grasp": "The robot is closing its gripper to secure a grasp on the object.",
    "lift": "The robot is lifting the grasped object upward from the surface.",
    "move": "The robot is transporting the grasped object toward the target location.",
    "place": "The robot is placing the object down at the target location and releasing.",
}

CONTACT_PHRASES = {
    "no_contact": "there is no physical contact between gripper and object",
    "contact": "the gripper is in contact with the object",
}

GRIPPER_PHRASES = {
    "open": "gripper is fully open",
    "closing": "gripper is actively closing",
    "closed": "gripper is fully closed around the object",
}

MOTION_PHRASES = {
    "still": "the object remains stationary",
    "lifted": "the object is being lifted off the surface",
    "moved": "the object is being moved laterally",
    "placed": "the object is being set down",
}

RELATION_PHRASES = {
    "farther": "the object is far from the target",
    "closer": "the object is moving closer to the target",
    "reached": "the object has reached the target location",
}


def generate_template_text(labels: dict[str, str]) -> str:
    """Deterministic template: same information as discrete labels, expressed in natural language."""
    contact = CONTACT_PHRASES.get(labels.get("contact_state", ""), "unknown contact state")
    gripper = GRIPPER_PHRASES.get(labels.get("gripper_state", ""), "unknown gripper state")
    motion = MOTION_PHRASES.get(labels.get("object_motion", ""), "unknown object motion")
    relation = RELATION_PHRASES.get(labels.get("target_relation", ""), "unknown target relation")

    return (
        f"During this manipulation step, {contact}. "
        f"The {gripper}. Meanwhile, {motion}, and {relation}."
    )


def generate_enriched_text(labels: dict[str, str], stage: str | None = None) -> str:
    """Richer description that adds physical context and commonsense reasoning."""
    contact = labels.get("contact_state", "")
    gripper = labels.get("gripper_state", "")
    motion = labels.get("object_motion", "")
    relation = labels.get("target_relation", "")

    parts = []

    if stage and stage in STAGE_TEMPLATES:
        parts.append(STAGE_TEMPLATES[stage])

    if contact == "no_contact" and gripper == "open":
        parts.append(
            "The gripper is open and has not yet touched the object. "
            "This is a pre-contact phase where the arm is positioning itself."
        )
    elif contact == "contact" and gripper == "closing":
        parts.append(
            "The gripper has made contact and is actively closing to secure the object. "
            "Force is being applied to establish a stable grasp."
        )
    elif contact == "contact" and gripper == "closed":
        if motion == "lifted":
            parts.append(
                "The object is firmly grasped and being lifted vertically. "
                "The gripper maintains a stable hold while overcoming gravity."
            )
        elif motion == "moved":
            parts.append(
                "The object is securely held and being transported horizontally. "
                "The arm is moving the object through free space toward the goal."
            )
        elif motion == "placed":
            parts.append(
                "The object is being lowered and placed at the target. "
                "The gripper will release once the object is stably supported."
            )
        elif motion == "still":
            parts.append(
                "The object is grasped but not yet moving. "
                "The robot may be stabilizing its grip before the next action."
            )
    else:
        parts.append(generate_template_text(labels))

    if relation == "reached":
        parts.append("The object has arrived at its intended destination.")
    elif relation == "closer":
        parts.append("The object is getting closer to the target position.")
    elif relation == "farther":
        parts.append("The object is still far from the target.")

    return " ".join(parts)


def generate_llm_prompt(labels: dict[str, str], stage: str | None = None) -> str:
    """Generate a prompt for an LLM to produce an enriched process description.

    Use this to batch-generate enriched texts offline via GPT/Claude API.
    """
    label_str = ", ".join(f"{k}={v}" for k, v in labels.items() if k != "stage_label")
    stage_str = f" The current manipulation stage is '{stage}'." if stage else ""

    return (
        f"You are describing a short-horizon robot manipulation step for a research dataset. "
        f"The observed state attributes are: {label_str}.{stage_str} "
        f"Write a concise (2-3 sentence) physical description of what is happening, "
        f"including the spatial relationship between gripper and object, "
        f"the forces involved, and what will likely happen next. "
        f"Be specific and grounded in robotics terminology."
    )
