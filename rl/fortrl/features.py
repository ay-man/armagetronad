"""Turns raw bridge messages into fixed size, egocentric observation vectors.

Everything is expressed relative to the agent's position and heading so that
a skill learned on one side of the map transfers to the other.
"""

from dataclasses import dataclass
import math

import numpy as np

# gSensorWallType values reported by the bridge (0 = nothing hit)
WALL_TYPES = (1, 2, 3, 4)  # rim, enemy, teammate, self

# discrete action table: (turn, brake); turn -1 = left, 1 = right
ACTIONS = (
    (0, 0), (-1, 0), (1, 0),
    (0, 1), (-1, 1), (1, 1),
)


@dataclass(frozen=True)
class ObsSpec:
    rays: int = 16
    max_enemies: int = 1
    max_mates: int = 0
    max_own_zones: int = 1
    max_enemy_zones: int = 1
    arena_scale: float = 500.0
    speed_scale: float = 30.0
    time_scale: float = 60.0

    def layout(self):
        """(name, start, slot width, slots) of each block, in vector order."""
        blocks = [("rays", 1 + len(WALL_TYPES), self.rays), ("self", 7, 1),
                  ("enemies", 8, self.max_enemies), ("mates", 8, self.max_mates),
                  ("own_zones", 8, self.max_own_zones), ("enemy_zones", 8, self.max_enemy_zones)]
        out, start = [], 0
        for name, width, slots in blocks:
            out.append((name, start, width, slots))
            start += width * slots
        return out

    @property
    def size(self):
        return (self.rays * (1 + len(WALL_TYPES)) + 7
                + 8 * (self.max_enemies + self.max_mates)
                + 8 * (self.max_own_zones + self.max_enemy_zones))


def _local(agent, x, y):
    """Coordinates of (x, y) in the agent's frame: (forward, left)."""
    vx, vy = x - agent["x"], y - agent["y"]
    dx, dy = agent["dx"], agent["dy"]
    return vx * dx + vy * dy, dx * vy - dy * vx


def _cycle_features(agent, other, spec):
    f, l = _local(agent, other["x"], other["y"])
    s = spec.arena_scale
    cos_h = other["dx"] * agent["dx"] + other["dy"] * agent["dy"]
    sin_h = agent["dx"] * other["dy"] - agent["dy"] * other["dx"]
    return [1.0, float(other["alive"]), f / s, l / s, math.hypot(f, l) / s, cos_h, sin_h,
            other.get("speed", 0.0) / spec.speed_scale]


def _zone_features(agent, zone, spec):
    f, l = _local(agent, zone["x"], zone["y"])
    s = spec.arena_scale
    dist = math.hypot(f, l)
    return [1.0, f / s, l / s, dist / s, zone["radius"] / s, zone["conquered"], float(zone["safe"]),
            float(dist < zone["radius"])]


def _nearest(agent, items, count, width, featurize):
    items = sorted(items, key=lambda o: (o["x"] - agent["x"]) ** 2 + (o["y"] - agent["y"]) ** 2)
    out = []
    for item in items[:count]:
        out += featurize(item)
    out += [0.0] * (width * (count - min(count, len(items))))
    return out


def encode(msg, agent, spec):
    """Observation vector for one steered agent of a step message."""
    if len(agent["rays"]) != spec.rays:
        raise ValueError(f"server sends {len(agent['rays'])} rays but ObsSpec expects {spec.rays}; "
                         "align RL_RAYS with the training config")
    ray_range = msg["ray_range"]
    out = []
    for dist, wall_type in agent["rays"]:
        out.append(dist / ray_range)
        out += [1.0 if wall_type == t else 0.0 for t in WALL_TYPES]

    out += [agent["speed"] / spec.speed_scale, agent["rubber"], agent["brake_reservoir"],
            float(agent["braking"]), float(agent["can_turn_left"]), float(agent["can_turn_right"]),
            min(msg["time"] / spec.time_scale, 2.0)]

    team = agent["team"]
    others = [c for c in msg["cycles"] if c["id"] != agent["id"] and "x" in c]
    enemies = [c for c in others if c["team"] != team]
    mates = [c for c in others if c["team"] == team]
    out += _nearest(agent, enemies, spec.max_enemies, 8, lambda c: _cycle_features(agent, c, spec))
    out += _nearest(agent, mates, spec.max_mates, 8, lambda c: _cycle_features(agent, c, spec))

    own = [z for z in msg["zones"] if z["team"] == team]
    hostile = [z for z in msg["zones"] if z["team"] != team and z["team"] >= 0]
    out += _nearest(agent, own, spec.max_own_zones, 8, lambda z: _zone_features(agent, z, spec))
    out += _nearest(agent, hostile, spec.max_enemy_zones, 8, lambda z: _zone_features(agent, z, spec))

    vec = np.asarray(out, dtype=np.float32)
    assert vec.shape == (spec.size,), (vec.shape, spec.size)
    return vec


class RunningNorm:
    """Running mean/variance normalizer (parallel Welford), saved with checkpoints."""

    def __init__(self, size, clip=10.0):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None]
        b_mean, b_var, b_count = batch.mean(0), batch.var(0), batch.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count, "clip": self.clip}

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
        self.clip = float(state["clip"])
