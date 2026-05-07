from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionBatch:
    text_features: np.ndarray
    delta_z: np.ndarray
    semantics: np.ndarray


def concat_features(batch: FusionBatch) -> np.ndarray:
    return np.concatenate(
        [batch.text_features, batch.delta_z, batch.semantics],
        axis=-1,
    )
