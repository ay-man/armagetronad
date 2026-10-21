"""Records full matches for the replay viewer.

    python -m fortrl.record configs/train_1v1_builtin.toml --rounds 3 --out replays.json \\
        --policy random --policy runs/m1_builtin/checkpoints/ckpt_000010.pt \\
        --policy runs/m1_builtin/checkpoints/latest.pt

Each policy plays team 0 against the game's built-in AI on a fresh server.
"""

import argparse
import json
import logging
from pathlib import Path

from .arena import Arena, Runner
from .factory import load_config, obs_spec, start_server
from .policy import PolicyBundle, PolicyController, RandomController
from .rewards import RewardConfig

KEEP_EVENTS = ("DEATH_", "BASEZONE_", "ROUND_WINNER")


def record_policy(cfg, policy, rounds, port, greedy=False):
    if policy == "random":
        controller, spec = RandomController(), obs_spec(cfg)
        controller.name = "policy"
    else:
        bundle = PolicyBundle.load(policy)
        controller, spec = PolicyController(bundle, "policy", greedy=greedy), bundle.spec

    server = start_server(cfg, 1, port, spec)
    arena = Arena(server, "record", lambda r: {0: controller})
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

    runner.observers.append(observe)
    try:
        runner.run(lambda r: len([x for x in r.records if x["result"] != "aborted"]) >= rounds)
    finally:
        runner.close()

    out = []
    for rec in runner.records:
        if rec["result"] == "aborted" or rec["round"] not in games:
            continue
        game = games[rec["round"]]
        out.append({"policy": policy, "result": rec["result"], "survival": rec["survival"],
                    "kills": rec["kills"], "reward": rec["reward"], **game})
    return out[:rounds]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--policy", action="append", required=True, help="checkpoint path or 'random'; repeatable")
    parser.add_argument("--label", action="append", help="display name per --policy")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--port", type=int, default=5200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    labels = args.label or [Path(p).stem for p in args.policy]
    if len(labels) != len(args.policy):
        parser.error("give one --label per --policy")

    cfg = load_config(args.config)
    result = []
    for i, (policy, label) in enumerate(zip(args.policy, labels)):
        games = record_policy(cfg, policy, args.rounds, args.port + i, args.greedy)
        for g in games:
            g["label"] = label
        result += games
        print(f"{label}: " + ", ".join(f"{g['result']} ({g['survival']:.0f}s)" for g in games))
    Path(args.out).write_text(json.dumps({"games": result}, separators=(",", ":")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
