"""Actor-critic network and the controllers that pick actions for steered cycles."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .features import ACTIONS, ObsSpec, RunningNorm


class ActorCritic(nn.Module):
    def __init__(self, obs_size, n_actions=len(ACTIONS), hidden=256):
        super().__init__()

        def mlp(out, gain):
            layers = [nn.Linear(obs_size, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                      nn.Linear(hidden, out)]
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, np.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(layers[-1].weight, gain)
            return nn.Sequential(*layers)

        self.actor = mlp(n_actions, 0.01)
        self.critic = mlp(1, 1.0)

    def forward(self, obs):
        return Categorical(logits=self.actor(obs)), self.critic(obs).squeeze(-1)


class PolicyBundle:
    """Network plus the observation normalizer and spec it was trained with."""

    def __init__(self, spec, hidden=256):
        self.spec = spec
        self.hidden = hidden
        self.model = ActorCritic(spec.size, hidden=hidden)
        self.norm = RunningNorm(spec.size)

    def act(self, obs_batch, greedy=False):
        """Returns (actions, log_probs, values) as numpy arrays."""
        x = torch.from_numpy(self.norm(np.asarray(obs_batch)))
        with torch.inference_mode():
            logp_all = torch.log_softmax(self.model.actor(x), -1)
            value = self.model.critic(x).squeeze(-1)
            action = logp_all.argmax(-1) if greedy else torch.multinomial(logp_all.exp(), 1).squeeze(-1)
            logp = logp_all.gather(-1, action[:, None]).squeeze(-1)
        return action.numpy(), logp.numpy(), value.numpy()

    def state_dict(self):
        return {"spec": self.spec.__dict__, "hidden": self.hidden, "model": self.model.state_dict(),
                "norm": self.norm.state_dict()}

    @classmethod
    def from_state_dict(cls, state):
        bundle = cls(ObsSpec(**state["spec"]), hidden=state["hidden"])
        bundle.model.load_state_dict(state["model"])
        bundle.norm.load_state_dict(state["norm"])
        return bundle

    @classmethod
    def transfer(cls, old, spec):
        """A bundle for a bigger observation layout that starts out acting exactly like `old`.

        Inputs that exist in both layouts keep their weights and normalizer statistics; new
        slots (more teammates, more enemies) start with zero input weights, so learning can
        give them meaning without first destroying the old skills.
        """
        if old.spec.rays != spec.rays:
            raise ValueError("ray count must match to transfer a policy")
        new = cls(spec, hidden=old.hidden)
        old_state, new_state = old.model.state_dict(), new.model.state_dict()
        cols_new, cols_old = [], []
        for (name, s_new, width, slots_new), (_, s_old, _, slots_old) in zip(spec.layout(), old.spec.layout()):
            for k in range(min(slots_new, slots_old) * width):
                cols_new.append(s_new + k)
                cols_old.append(s_old + k)
        for key, value in old_state.items():
            if key in ("actor.0.weight", "critic.0.weight"):
                w = torch.zeros_like(new_state[key])
                w[:, cols_new] = value[:, cols_old]
                new_state[key] = w
            else:
                new_state[key] = value.clone()
        new.model.load_state_dict(new_state)
        new.norm.mean[cols_new] = old.norm.mean[cols_old]
        new.norm.var[cols_new] = old.norm.var[cols_old]
        new.norm.count = old.norm.count
        return new

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_state_dict(state["policy"] if "policy" in state else state)


class Controller:
    """Chooses actions for a set of agents. learns=True marks the policy being trained."""

    learns = False
    name = "controller"

    def act(self, obs_batch):
        raise NotImplementedError


class RandomController(Controller):
    name = "random"

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def act(self, obs_batch):
        n = len(obs_batch)
        return self.rng.integers(len(ACTIONS), size=n), np.zeros(n), np.zeros(n)


class PolicyController(Controller):
    def __init__(self, bundle, name, learns=False, greedy=False):
        self.bundle = bundle
        self.name = name
        self.learns = learns
        self.greedy = greedy

    def act(self, obs_batch):
        return self.bundle.act(obs_batch, greedy=self.greedy)
