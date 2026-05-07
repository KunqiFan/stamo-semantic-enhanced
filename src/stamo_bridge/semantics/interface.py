from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEMANTIC_FIELDS = [
    "contact_state",
    "gripper_state",
    "object_motion",
    "target_relation",
]

SEMANTIC_LABELS = {
    "contact_state": ["no_contact", "contact"],
    "gripper_state": ["open", "closing", "closed"],
    "object_motion": ["still", "lifted", "moved", "placed"],
    "target_relation": ["farther", "closer", "reached"],
}


@dataclass
class CompactProcessSemantics:
    contact_state: str
    gripper_state: str
    object_motion: str
    target_relation: str

    def as_text(self) -> str:
        return (
            f"contact_state={self.contact_state}; "
            f"gripper_state={self.gripper_state}; "
            f"object_motion={self.object_motion}; "
            f"target_relation={self.target_relation}"
        )


def semantics_vectorize(semantics: CompactProcessSemantics) -> np.ndarray:
    vector: list[float] = []
    for field in SEMANTIC_FIELDS:
        current_value = getattr(semantics, field)
        for candidate in SEMANTIC_LABELS[field]:
            vector.append(1.0 if current_value == candidate else 0.0)
    return np.asarray(vector, dtype=np.float32)


def semantics_from_dict(values: dict[str, str]) -> CompactProcessSemantics:
    return CompactProcessSemantics(
        contact_state=values["contact_state"],
        gripper_state=values["gripper_state"],
        object_motion=values["object_motion"],
        target_relation=values["target_relation"],
    )
