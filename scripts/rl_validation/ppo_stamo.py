"""Minimal PPO for ManiSkill3 with StaMo encoder observations.

Trains a policy on PickCube-v1 or StackCube-v1 using StaMo-encoded
observations. The StaMo encoder is frozen; only the actor-critic MLP
is trained.

Usage:
    cd "stamo_pro - 副本 - 副本"
    py rl_validation/scripts/ppo_stamo.py \
        --task PickCube-v1 \
        --checkpoint StaMo/ckpts/maniskill_pickcube_v1_diffonly/5000 \
        --config rl_validation/configs/stamo_maniskill.yaml \
        --total_timesteps 500000 --seed 42 \
        --run_name pickcube_diffonly_s42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

import mani_skill.envs  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rl_validation.wrappers.stamo_encoder_wrapper import StaMoEncoderWrapper


# ── Actor-Critic ───────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.actor_logstd = nn.Parameter(-0.5 * torch.ones(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def get_action_and_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)


# ── Rollout buffer ─────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, act_dim), dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.ptr = 0
        self.n_steps = n_steps

    def store(self, obs, action, log_prob, reward, done, value):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_val = last_value
            else:
                next_val = self.values[t + 1]
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * nonterminal - self.values[t]
            advantages[t] = last_adv = delta + gamma * gae_lambda * nonterminal * last_adv
        self.advantages = advantages
        self.returns = advantages + self.values

    def reset(self):
        self.ptr = 0


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(env, agent, n_episodes: int = 50, device: str = "cuda"):
    successes = []
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            action_np = action[0].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_return += float(reward)
            done = terminated or truncated
        success = info.get("success", False)
        if hasattr(success, "item"):
            success = bool(success.item())
        successes.append(success)
        returns.append(ep_return)
    return float(np.mean(successes)), float(np.mean(returns))


# ── PPO update ─────────────────────────────────────────────────────────────

def ppo_update(agent, optimizer, buf, clip_eps, vf_coef, ent_coef,
               n_epochs, batch_size, device):
    obs_t = torch.from_numpy(buf.obs).float().to(device)
    act_t = torch.from_numpy(buf.actions).float().to(device)
    old_lp = torch.from_numpy(buf.log_probs).float().to(device)
    adv_t = torch.from_numpy(buf.advantages).float().to(device)
    ret_t = torch.from_numpy(buf.returns).float().to(device)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(obs_t)
    total_pg_loss = 0.0
    total_vf_loss = 0.0
    total_ent = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n:
                break
            idx = indices[start:end]
            b_obs = obs_t[idx]
            b_act = act_t[idx]
            b_old_lp = old_lp[idx]
            b_adv = adv_t[idx]
            b_ret = ret_t[idx]

            _, new_lp, entropy, values = agent.get_action_and_value(b_obs, b_act)
            ratio = (new_lp - b_old_lp).exp()
            pg_loss1 = -b_adv * ratio
            pg_loss2 = -b_adv * ratio.clamp(1 - clip_eps, 1 + clip_eps)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            vf_loss = 0.5 * (values - b_ret).pow(2).mean()
            ent_loss = -entropy.mean()

            loss = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_ent += entropy.mean().item()
            n_updates += 1

    return {
        "pg_loss": total_pg_loss / max(n_updates, 1),
        "vf_loss": total_vf_loss / max(n_updates, 1),
        "entropy": total_ent / max(n_updates, 1),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["PickCube-v1", "StackCube-v1", "PushCube-v1"])
    parser.add_argument("--checkpoint", required=True, help="StaMo checkpoint dir")
    parser.add_argument("--config", required=True, help="StaMo config yaml")
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="ppo_run")
    # PPO hyperparams
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results_dir = ROOT / "rl_validation" / "results" / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task: {args.task}, Seed: {args.seed}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Total timesteps: {args.total_timesteps}")

    # Create environment with StaMo encoder
    env = gym.make(args.task, obs_mode="state_dict", control_mode=args.control_mode)
    env = StaMoEncoderWrapper(env, args.checkpoint, args.config, device=args.device)

    # Share encoder model with eval env to save VRAM
    eval_env = gym.make(args.task, obs_mode="state_dict", control_mode=args.control_mode)
    eval_env = StaMoEncoderWrapper(eval_env, args.checkpoint, args.config,
                                   device=args.device, shared_model=env.model)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

    agent = ActorCritic(obs_dim, act_dim).to(args.device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    buf = RolloutBuffer(args.n_steps, obs_dim, act_dim)
    log = []

    obs, _ = env.reset(seed=args.seed)
    global_step = 0
    n_updates = 0
    episode_returns = []
    ep_return = 0.0
    t_start = time.time()
    last_eval_step = 0

    print(f"\nTraining...")

    while global_step < args.total_timesteps:
        # Collect rollout
        buf.reset()
        encoder_time = 0.0
        for _ in range(args.n_steps):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(args.device)
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs_t)
            action_np = action[0].cpu().numpy()
            log_prob_np = log_prob.item()
            value_np = value.item()

            t_enc = time.time()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            encoder_time += time.time() - t_enc

            done = terminated or truncated
            buf.store(obs, action_np, log_prob_np, float(reward), float(done), value_np)
            obs = next_obs
            ep_return += float(reward)
            global_step += 1

            if done:
                episode_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()

        # Compute GAE
        with torch.no_grad():
            last_val = agent.get_value(
                torch.from_numpy(obs).float().unsqueeze(0).to(args.device)).item()
        buf.compute_gae(last_val, args.gamma, args.gae_lambda)

        # PPO update
        stats = ppo_update(agent, optimizer, buf, args.clip_eps, args.vf_coef,
                           args.ent_coef, args.n_epochs, args.batch_size, args.device)
        n_updates += 1

        # Log
        elapsed = time.time() - t_start
        sps = global_step / elapsed
        recent_returns = episode_returns[-10:] if episode_returns else [0]
        print(f"  step {global_step}/{args.total_timesteps}  "
              f"return={np.mean(recent_returns):.2f}  "
              f"pg={stats['pg_loss']:.4f}  vf={stats['vf_loss']:.4f}  "
              f"ent={stats['entropy']:.3f}  "
              f"enc_t={encoder_time:.1f}s  sps={sps:.0f}", flush=True)

        # Evaluate
        if global_step - last_eval_step >= args.eval_interval:
            last_eval_step = global_step
            success_rate, mean_return = evaluate(
                eval_env, agent, args.eval_episodes, args.device)
            entry = {
                "step": global_step,
                "success_rate": success_rate,
                "eval_return": mean_return,
                "train_return": float(np.mean(recent_returns)),
                "wall_clock": elapsed,
                "encoder_time_per_rollout": encoder_time,
            }
            log.append(entry)
            print(f"  ** EVAL step={global_step}: success={success_rate:.1%}, "
                  f"return={mean_return:.2f}", flush=True)

            with open(results_dir / "log.json", "w") as f:
                json.dump(log, f, indent=2)

            torch.save(agent.state_dict(), results_dir / "agent.pt")

    # Final eval
    success_rate, mean_return = evaluate(eval_env, agent, args.eval_episodes, args.device)
    entry = {
        "step": global_step,
        "success_rate": success_rate,
        "eval_return": mean_return,
        "train_return": float(np.mean(episode_returns[-10:])),
        "wall_clock": time.time() - t_start,
        "encoder_time_per_rollout": 0.0,
    }
    log.append(entry)
    print(f"\nFinal: success={success_rate:.1%}, return={mean_return:.2f}")

    with open(results_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)
    torch.save(agent.state_dict(), results_dir / "agent.pt")

    with open(results_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    env.close()
    eval_env.close()
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
