"""Proximal Policy Optimization update (clipped objective, GAE advantages)."""

from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn


@dataclass
class PPOConfig:
    rollout_steps: int = 4096
    epochs: int = 4
    minibatch: int = 512
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    target_kl: float = 0.03

    @classmethod
    def from_dict(cls, d):
        unknown = set(d) - set(asdict(cls()))
        if unknown:
            raise ValueError(f"unknown ppo settings: {sorted(unknown)}")
        return cls(**d)


def ppo_update(bundle, optimizer, batch, cfg):
    """One PPO update on a batch from Runner.take_batch. Returns diagnostics."""
    model = bundle.model
    obs = torch.as_tensor(bundle.norm(batch["obs"]))
    act = torch.as_tensor(batch["act"])
    old_logp = torch.as_tensor(batch["logp"])
    adv = torch.as_tensor(batch["adv"])
    ret = torch.as_tensor(batch["ret"])
    old_val = torch.as_tensor(batch["val"])

    n = len(obs)
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}
    stopped_early = False
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.minibatch):
            idx = perm[start:start + cfg.minibatch]
            dist, value = model(obs[idx])
            logp = dist.log_prob(act[idx])
            ratio = torch.exp(logp - old_logp[idx])

            a = adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8) if len(a) > 1 else a
            pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()

            v_clipped = old_val[idx] + (value - old_val[idx]).clamp(-cfg.clip, cfg.clip)
            v_loss = 0.5 * torch.max((value - ret[idx]) ** 2, (v_clipped - ret[idx]) ** 2).mean()
            entropy = dist.entropy().mean()

            loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = logp - old_logp[idx]
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
            stats["policy_loss"].append(pg_loss.item())
            stats["value_loss"].append(v_loss.item())
            stats["entropy"].append(entropy.item())
            stats["approx_kl"].append(approx_kl)
            stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
        if cfg.target_kl and np.mean(stats["approx_kl"][-max(1, n // cfg.minibatch):]) > cfg.target_kl:
            stopped_early = True
            break

    var_ret = np.var(batch["ret"])
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["explained_variance"] = float(1 - np.var(batch["ret"] - batch["val"]) / var_ret) if var_ret > 0 else 0.0
    out["kl_stop"] = int(stopped_early)
    return out
