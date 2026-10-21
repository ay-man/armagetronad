"""Evaluates a policy against fixed opponents in fresh, non-learning matches.

    python -m fortrl.evaluate configs/train_1v1_builtin.toml --policy runs/first/checkpoints/latest.pt
    python -m fortrl.evaluate configs/train_1v1_builtin.toml --policy random --opponents builtin
    python -m fortrl.evaluate configs/train_1v1_builtin.toml --policy runs/first/checkpoints/latest.pt \\
        --opponents ckpt:runs/first/checkpoints/ckpt_000010.pt
"""

import argparse
import json
import logging

from .evaluation import evaluate
from .factory import load_config, obs_spec
from .policy import PolicyBundle, PolicyController, RandomController
from .stats import format_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--policy", required=True, help="checkpoint path, or 'random' for the random baseline")
    parser.add_argument("--opponents", nargs="+", default=["builtin", "random"],
                        help="builtin, random, ckpt:<path>")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--greedy", action="store_true", help="take the most likely action instead of sampling")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write results as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    cfg = load_config(args.config)
    if args.policy == "random":
        controller = RandomController()
        controller.name = "eval"
        spec = obs_spec(cfg)
    else:
        bundle = PolicyBundle.load(args.policy)
        controller = PolicyController(bundle, "eval", greedy=args.greedy)
        spec = bundle.spec

    results = evaluate(cfg, controller, args.opponents, args.rounds, spec, args.port, args.seed)
    for opp, s in results.items():
        print(format_summary(f"{args.policy} vs {opp}", s))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"policy": args.policy, "greedy": args.greedy, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
