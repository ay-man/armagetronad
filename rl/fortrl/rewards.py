"""Reward computation from consecutive bridge messages.

The outcome (win/loss/draw) is the signal that defines success. The other
terms are optional shaping that make the sparse outcome easier to discover;
each is reported separately so its influence on learning can be inspected.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class RewardConfig:
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    death: float = -0.5
    kill: float = 0.5
    # per unit of conquest progress (0..1) on enemy / own fortress
    enemy_conquest: float = 1.0
    own_conquest: float = -1.0
    # per second alive
    alive: float = 0.0

    @classmethod
    def from_dict(cls, d):
        unknown = set(d) - set(asdict(cls()))
        if unknown:
            raise ValueError(f"unknown reward settings: {sorted(unknown)}")
        return cls(**d)


@dataclass
class RewardBreakdown:
    parts: dict = field(default_factory=dict)

    def add(self, name, value):
        if value:
            self.parts[name] = self.parts.get(name, 0.0) + value

    @property
    def total(self):
        return sum(self.parts.values())


def _progress(zone):
    return 1.0 if not zone["safe"] else zone["conquered"]


def conquest_delta(prev, cur, team):
    """Change in conquest progress of (enemy, own) fortresses; a vanished zone counts as conquered."""
    current = {(z["team"], round(z["x"]), round(z["y"])): z for z in cur["zones"]}
    enemy = own = 0.0
    for z in prev["zones"]:
        if z["team"] < 0:
            continue
        match = current.get((z["team"], round(z["x"]), round(z["y"])))
        delta = (_progress(match) if match else 1.0) - _progress(z)
        if z["team"] == team:
            own += delta
        else:
            enemy += delta
    return enemy, own


def parse_events(events):
    """Splits ladderlog lines into (kind, fields)."""
    return [(e.split(" ", 1)[0], e.split(" ")[1:]) for e in events]


def step_reward(cfg, prev, cur, agent, prev_alive, now_alive, out=None):
    """Reward for one agent between two messages (cur may be a round_end message)."""
    out = RewardBreakdown() if out is None else out
    team, name = agent["team"], agent["name"]

    enemy, own = conquest_delta(prev, cur, team)
    out.add("enemy_conquest", cfg.enemy_conquest * enemy)
    out.add("own_conquest", cfg.own_conquest * own)

    if prev_alive:
        out.add("alive", cfg.alive * max(0.0, cur["time"] - prev["time"]))
    if prev_alive and not now_alive:
        out.add("death", cfg.death)

    for kind, fields in parse_events(cur["events"]):
        if kind == "DEATH_FRAG" and len(fields) >= 2 and fields[1] == name and fields[0] != name:
            out.add("kill", cfg.kill)
    return out


def outcome(cur, team):
    """'win', 'loss', 'draw' or 'aborted' (round cut short undecided) for a round_end message."""
    winner = cur["winner"]
    if winner > 0:
        return "win" if winner - 1 == team else "loss"
    return "draw" if winner < 0 else "aborted"


def outcome_reward(cfg, result, out=None):
    out = RewardBreakdown() if out is None else out
    out.add("outcome", {"win": cfg.win, "loss": cfg.loss, "draw": cfg.draw, "aborted": 0.0}[result])
    return out
