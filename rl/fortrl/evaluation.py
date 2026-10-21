"""Separate evaluation matches: fresh servers, no learning, fixed opponents."""

import logging

from .arena import Runner
from .factory import eval_arena
from .rewards import RewardConfig
from .stats import summarize

log = logging.getLogger(__name__)


def evaluate(cfg, controller, opponents, rounds, spec, base_port, seed=0, servers=3):
    """Plays `rounds` rounds of `controller` against each opponent kind, spread over `servers` servers.

    Opponent kinds: 'builtin' (the game's AI), 'random', 'ckpt:<path>'.
    Returns {opponent: summary} for the evaluated controller's rounds only.
    """
    reward_cfg = RewardConfig.from_dict(cfg.get("reward", {}))
    per_server = -(-rounds // servers)
    results = {}
    for i, opponent in enumerate(opponents):
        arenas, runner = [], None
        try:
            for j in range(servers):
                arenas.append(eval_arena(cfg, opponent, controller, spec, base_port + i * servers + j,
                                         seed * 1000 + i * servers + j))
            runner = Runner(arenas, spec, reward_cfg)

            # each server contributes its first rounds; taking whichever rounds finish first
            # overall would over-sample short rounds
            def per_arena(r):
                out = {a.name: [] for a in arenas}
                for x in r.records:
                    if x["controller"] == controller.name and x["result"] != "aborted":
                        out[x["arena"]].append(x)
                return out

            runner.run(lambda r: all(len(v) >= per_server for v in per_arena(r).values()))
            chosen = [x for v in per_arena(runner).values() for x in v[:per_server]]
            results[opponent] = summarize(chosen)
        finally:
            if runner is not None:
                runner.selector.close()
            for arena in arenas:
                arena.server.close()
    return results
