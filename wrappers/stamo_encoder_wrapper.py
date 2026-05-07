"""Gym observation wrapper: ManiSkill3 env → StaMo encoder → feature vector.

Wraps a ManiSkill3 environment so that each observation includes a StaMo
feature vector extracted from the rendered RGB frame, concatenated with
the low-dimensional proprioceptive state (agent + extra).

The StaMo encoder is frozen (no gradients).
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
STAMO_ROOT = ROOT / "StaMo"
if str(STAMO_ROOT) not in sys.path:
    sys.path.insert(0, str(STAMO_ROOT))

from omegaconf import OmegaConf
from stamo.renderer.model.renderer import RenderNet


class StaMoEncoderWrapper(gym.ObservationWrapper):
    """Replaces image observations with StaMo feature vectors.

    Observation space becomes a flat vector:
        stamo_features (2560D) + full state (agent + extra)
    """

    def __init__(self, env: gym.Env, checkpoint_path: str, config_path: str,
                 device: str = "cuda", shared_model: RenderNet | None = None):
        super().__init__(env)
        self.device = device

        args = OmegaConf.load(config_path)
        self._img_size = args.data.img_size

        if shared_model is not None:
            self.model = shared_model
        else:
            args.deepspeed = False
            self.model = RenderNet(args)
            self.model.load_checkpoint(checkpoint_path)
            self.model = self.model.to(device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

        # ImageNet normalization constants
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        # Determine feature dim
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        dummy = (dummy - self._mean) / self._std
        with torch.no_grad():
            embeds, pooled = self.model.encode(dummy)
        self.feature_dim = embeds.shape[1] * embeds.shape[2] + pooled.shape[1]

        # Determine state dim from a reset
        self._render_cam = self._get_render_camera()
        obs, _ = self.env.reset()
        state = self._flatten_state(obs)
        self._state_dim = len(state)

        total_dim = self.feature_dim + self._state_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

    def _get_render_camera(self):
        return self.env.unwrapped.scene.sensors["base_camera"].camera._render_cameras[0]

    def _flatten_state(self, obs) -> np.ndarray:
        parts = []
        if "agent" in obs:
            for v in obs["agent"].values():
                t = v[0] if v.dim() > 1 else v
                parts.append(t.cpu().numpy().flatten())
        if "extra" in obs:
            for v in obs["extra"].values():
                t = v[0] if v.dim() > 0 and v.shape[0] == 1 else v
                parts.append(t.cpu().numpy().flatten())
        return np.concatenate(parts).astype(np.float32)

    def _capture_rgb(self) -> np.ndarray:
        self.env.unwrapped.scene.update_render()
        self._render_cam.take_picture()
        rgba = self._render_cam.get_picture("Color")
        rgb = (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
        if rgb.shape[0] != self._img_size or rgb.shape[1] != self._img_size:
            rgb = np.array(
                Image.fromarray(rgb).resize(
                    (self._img_size, self._img_size), Image.LANCZOS))
        return rgb

    @torch.no_grad()
    def _encode_image(self, rgb: np.ndarray) -> np.ndarray:
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        img = (img - self._mean) / self._std
        embeds, pooled = self.model.encode(img)
        embeds_flat = embeds.reshape(1, -1)
        features = torch.cat([embeds_flat, pooled], dim=1)
        return features[0].float().cpu().numpy()

    def observation(self, obs):
        rgb = self._capture_rgb()
        stamo_feat = self._encode_image(rgb)
        state = self._flatten_state(obs)
        return np.concatenate([stamo_feat, state]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._render_cam = self._get_render_camera()
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), float(reward), bool(terminated), bool(truncated), info
