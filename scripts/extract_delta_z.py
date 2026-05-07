from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STAMO_ROOT = ROOT / "StaMo"
if str(STAMO_ROOT) not in sys.path:
    sys.path.insert(0, str(STAMO_ROOT))

from stamo.renderer.model.renderer import RenderNet  # noqa: E402


def load_manifest(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class StaMoFeatureExtractor:
    def __init__(self, config_path: Path, checkpoint_dir: Path | None = None, device: str | None = None) -> None:
        args = OmegaConf.load(config_path)
        args.deepspeed = False
        args.resume = checkpoint_dir is not None
        args.resume_path = str(checkpoint_dir) if checkpoint_dir is not None else ""
        self.model = RenderNet(args)
        if checkpoint_dir is not None:
            self.model.load_checkpoint(str(checkpoint_dir))

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.image_size = args.data.img_size
        self.to_tensor = T.Compose(
            [
                T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ]
        )

    def _load_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        tensor = self.to_tensor(image).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def extract_pair(self, image_t: str, image_tp: str) -> dict[str, np.ndarray]:
        start = self._load_image(image_t)
        end = self._load_image(image_tp)

        start_inputs = self.model.projector_feature_extractor(start)
        end_inputs = self.model.projector_feature_extractor(end)

        z_t, pooled_t = self.model.encode(start_inputs)
        z_tp, pooled_tp = self.model.encode(end_inputs)

        delta_z = z_tp - z_t
        delta_pooled = pooled_tp - pooled_t

        return {
            "z_t": z_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "z_tp": z_tp.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "delta_z": delta_z.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "pooled_t": pooled_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "pooled_tp": pooled_tp.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "delta_pooled": delta_pooled.squeeze(0).detach().cpu().numpy().astype(np.float32),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--stamo_config", type=Path, default=ROOT / "StaMo" / "configs" / "toy.yaml")
    parser.add_argument("--checkpoint_dir", type=Path, default=ROOT / "StaMo" / "ckpts" / "toy_debug" / "4")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    extractor = StaMoFeatureExtractor(
        config_path=args.stamo_config,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for row in load_manifest(args.manifest):
        features = extractor.extract_pair(row["image_t"], row["image_tp"])
        out_path = args.out_dir / f'{row["sample_id"]}.npz'
        np.savez_compressed(out_path, **features)

    print(f"Saved StaMo features to {args.out_dir}")


if __name__ == "__main__":
    main()
