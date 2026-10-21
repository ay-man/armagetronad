"""Records full matches for the replay viewer.

    python -m fortrl.record configs/train_1v1_builtin.toml --rounds 3 --out replays.json \\
        --policy random --policy runs/m1_builtin/checkpoints/ckpt_000010.pt \\
        --policy runs/m1_builtin/checkpoints/latest.pt

Each policy drives every cycle of team 0 against the game's built-in AI on a fresh server.
Results are from team 0's point of view.
"""

import argparse
import json
import logging
from pathlib import Path

from .arena import Arena, Runner
from .factory import fixed_opponent, load_config, obs_spec, start_server
from .policy import PolicyBundle, PolicyController, RandomController
from .rewards import RewardConfig

KEEP_EVENTS = ("DEATH_", "BASEZONE_", "ROUND_WINNER")


def record_policy(cfg, policy, rounds, port, greedy=False, opponent="builtin", label="", opponent_label=""):
    if policy in ("random", "builtin"):
        controller, spec = RandomController(), obs_spec(cfg)
        controller.name = "policy"
    else:
        bundle = PolicyBundle.load(policy)
        controller, spec = PolicyController(bundle, "policy", greedy=greedy), bundle.spec

    # with 'builtin' nothing is steered: the game's AI drives both teams
    if opponent == "builtin":
        server = start_server(cfg, 0 if policy == "builtin" else 1, port, spec)
        arena = Arena(server, "record", lambda r: {0: controller})
        team_labels = ["built-in AI" if policy == "builtin" else label or "learning bot", "built-in AI"]
    else:
        # a frozen checkpoint on team 1; both teams are steered from here
        server = start_server(cfg, 2, port, spec)
        other = fixed_opponent(opponent)
        arena = Arena(server, "record", lambda r: {0: controller, 1: other})
        team_labels = [label or "learning bot", opponent_label or other.name]
    runner = Runner([arena], spec, RewardConfig.from_dict(cfg.get("reward", {})))
    games = {}

    def observe(idx, msg, actions):
        if msg is None:
            frame = games[arena.round]["frames"][-1]
            frame["a"] = {str(pid): list(a) for pid, a in actions.items()}
            return
        game = games.setdefault(msg["round"], {"frames": [], "names": {}})
        for c in msg["cycles"]:
            game["names"][str(c["id"])] = {"name": c["name"], "team": c["team"],
                                          "steered": any(a["id"] == c["id"] for a in msg["agents"])}
        game["frames"].append({
            "t": round(msg["time"], 2),
            "c": [[c["id"], c["team"], round(c["x"], 1), round(c["y"], 1), c["alive"],
                   round(c["dx"], 2), round(c["dy"], 2)]
                  for c in msg["cycles"] if "x" in c],
            "z": [[round(z["x"], 1), round(z["y"], 1), round(z["radius"], 1), round(z["conquered"], 3),
                   z["safe"], z["team"]] for z in msg["zones"]],
            "e": [e for e in msg["events"] if e.startswith(KEEP_EVENTS)],
            "end": msg["type"] == "round_end",
        })
        game["team_labels"] = team_labels
        if msg["type"] == "round_end":
            game["winner"] = msg["winner"]

    runner.observers.append(observe)

    def decided(r):
        return [k for k, g in games.items() if g.get("winner", 0) != 0]

    try:
        runner.run(lambda r: len(decided(r)) >= rounds)
    finally:
        runner.close()

    out = []
    for k in decided(runner)[:rounds]:
        game = games[k]
        mine = [x for x in runner.records if x["round"] == k]
        survival = [x["survival"] for x in mine if x["survival"] is not None]
        result = "win" if game["winner"] == 1 else "loss" if game["winner"] > 1 else "draw"
        out.append({"policy": policy, "result": result, "agents": len(mine),
                    "survival": sum(survival) / len(survival) if survival else game["frames"][-1]["t"],
                    "kills": sum(x["kills"] for x in mine), **game})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--policy", action="append", required=True, help="checkpoint path, 'random', or 'builtin' (game AI on both teams); repeatable")
    parser.add_argument("--label", action="append", help="display name per --policy")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--opponent", default="builtin", help="'builtin' or ckpt:<path> on team 1")
    parser.add_argument("--opponent-label", default="", help="display name of the opponent team")
    parser.add_argument("--team-label", action="append", help="display name of team 0 per --policy")
    parser.add_argument("--port", type=int, default=5200)
    parser.add_argument("--arena-size", type=float, default=500, help="edge length of the square arena")
    parser.add_argument("--speed", type=float, default=4, help="default playback speed in the viewer")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    labels = args.label or [Path(p).stem for p in args.policy]
    if len(labels) != len(args.policy):
        parser.error("give one --label per --policy")

    cfg = load_config(args.config)
    result = []
    for i, (policy, label) in enumerate(zip(args.policy, labels)):
        team_label = args.team_label[i] if args.team_label else ""
        games = record_policy(cfg, policy, args.rounds, args.port + i, args.greedy, args.opponent, team_label,
                              args.opponent_label)
        for g in games:
            g["label"] = label
        result += games
        print(f"{label}: " + ", ".join(f"{g['result']} ({g['survival']:.0f}s)" for g in games))
    Path(args.out).write_text(json.dumps({"games": result, "arena": args.arena_size, "speed": args.speed},
                                         separators=(",", ":")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
