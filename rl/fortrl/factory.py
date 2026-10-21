"""Builds servers, arenas and opponent schedules from a training config."""

from dataclasses import asdict
import random
import tomllib
from pathlib import Path

from .arena import Arena
from .features import ObsSpec
from .policy import PolicyBundle, PolicyController, RandomController
from .server import GameServer, default_binary

RL_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = str(Path(path).resolve())
    return cfg


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else RL_ROOT / p


def obs_spec(cfg):
    return ObsSpec(**cfg.get("obs", {}))


def server_settings(spec):
    return {"RL_RAYS": spec.rays}


def start_server(cfg, steered_teams, port, spec):
    game = cfg.get("game", {})
    binary = game.get("server") or default_binary()
    return GameServer(binary, resolve(game.get("config", "configs/fort_1v1.cfg")), steered_teams, port,
                      extra_settings=server_settings(spec)).start()


class SnapshotPool:
    """Frozen earlier versions of the learner, used as opponents."""

    def __init__(self, directory, max_size=20):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._cache = {}

    def paths(self):
        return sorted(self.directory.glob("*.pt"))

    def add(self, bundle, tag):
        import torch
        torch.save({"policy": bundle.state_dict()}, self.directory / f"{tag}.pt")
        for old in self.paths()[:-self.max_size]:
            old.unlink()
            self._cache.pop(old, None)

    def sample(self, rng):
        paths = self.paths()
        if not paths:
            return None
        path = rng.choice(paths)
        if path not in self._cache:
            self._cache[path] = PolicyController(PolicyBundle.load(path), f"pool:{path.stem}")
        return self._cache[path]


class OpponentSchedule:
    """Per round controller assignment for an arena.

    With one steered team the learner always drives team 0 against the built-in AI.
    With two steered teams the learner takes a random side and the other side is drawn
    from the weighted opponent mix: random, self (the learner itself), pool (old snapshot).
    """

    def __init__(self, steered_teams, mix, learner, pool, rng):
        self.steered_teams = steered_teams
        self.mix = mix
        self.learner = learner
        self.pool = pool
        self.rng = rng
        self.random = RandomController()
        if steered_teams == 1 and set(mix) - {"builtin"}:
            raise ValueError("arenas with steered_teams = 1 can only play against the builtin AI")
        if steered_teams == 2 and "builtin" in mix:
            raise ValueError("builtin opponents need steered_teams = 1")
        if steered_teams not in (1, 2):
            raise ValueError("only 1 vs 1 arenas are supported so far")

    def __call__(self, round_number):
        if self.steered_teams == 1:
            return {0: self.learner}
        kinds, weights = zip(*self.mix.items())
        kind = self.rng.choices(kinds, weights)[0]
        if kind == "random":
            opponent = self.random
        elif kind == "self":
            opponent = self.learner
        elif kind == "pool":
            opponent = self.pool.sample(self.rng) or self.learner
        elif kind.startswith("ckpt:"):
            opponent = fixed_opponent(kind)
        else:
            raise ValueError(f"unknown opponent kind {kind!r}")
        side = self.rng.randrange(2)
        return {side: self.learner, 1 - side: opponent}


_fixed = {}


def fixed_opponent(kind):
    path = kind[len("ckpt:"):]
    if path not in _fixed:
        _fixed[path] = PolicyController(PolicyBundle.load(path), f"ckpt:{Path(path).stem}")
    return _fixed[path]


def make_arenas(cfg, learner, pool, spec, base_port, seed):
    arenas = []
    port = base_port
    rng = random.Random(seed)
    for group in cfg["arenas"]:
        for i in range(group.get("count", 1)):
            steered = group["steered_teams"]
            server = start_server(cfg, steered, port, spec)
            port += 1
            schedule = OpponentSchedule(steered, group["opponents"], learner, pool,
                                        random.Random(rng.random()))
            arenas.append(Arena(server, f"{group['name']}#{i}", schedule))
    return arenas


def eval_arena(cfg, opponent, controller, spec, port, seed):
    """An arena where `controller` plays every round against one fixed opponent kind."""
    if opponent == "builtin":
        server = start_server(cfg, 1, port, spec)
        return Arena(server, f"eval_builtin@{port}", lambda r: {0: controller})
    server = start_server(cfg, 2, port, spec)
    rng = random.Random(seed)
    other = RandomController() if opponent == "random" else fixed_opponent(opponent)

    def assign(r):
        side = rng.randrange(2)
        return {side: controller, 1 - side: other}
    return Arena(server, f"eval_{opponent}@{port}", assign)


def describe(spec):
    return asdict(spec)
