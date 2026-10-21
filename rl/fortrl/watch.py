"""Runs a real-time server where a trained policy drives every AI cycle.

Connect with a normal Armagetron Advanced client to watch (spectator mode) or to play
against it yourself:

    python -m fortrl.watch configs/train_1v1_builtin.toml --policy runs/m1_builtin/checkpoints/latest.pt
    # then in the game client: Play Game -> Multiplayer -> Custom Connect -> <this machine>:4534

With --vs-builtin, team 0 is the policy and the other team the game's own AI.
"""

import argparse
import logging

from .arena import Arena, Runner
from .factory import load_config, resolve
from .policy import PolicyBundle, PolicyController
from .rewards import RewardConfig
from .server import GameServer, default_binary
from .stats import format_summary, summarize


class _Everyone(dict):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def get(self, team, default=None):
        return self.controller


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--port", type=int, default=4534)
    parser.add_argument("--vs-builtin", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_config(args.config)
    bundle = PolicyBundle.load(args.policy)
    controller = PolicyController(bundle, "policy", greedy=args.greedy)
    game = cfg.get("game", {})
    server = GameServer(game.get("server") or default_binary(), resolve(game.get("config", "configs/fort_1v1.cfg")),
                        steered_teams=1 if args.vs_builtin else -1, port=args.port,
                        extra_settings={"RL_RAYS": bundle.spec.rays, "SIMULATION_FIXED_STEP": 0,
                                        "TALK_TO_MASTER": 0},
                        connect_timeout=120).start()
    print(f"server running on port {args.port}; server log: {server.log_path}", flush=True)
    arena = Arena(server, "watch", lambda r: _Everyone(controller))
    runner = Runner([arena], bundle.spec, RewardConfig.from_dict(cfg.get("reward", {})))
    try:
        while True:
            runner.run(lambda r: len(r.records) >= 10)
            print(format_summary("last 10 cycle-rounds of the policy", summarize(runner.records)))
            runner.records = []
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()


if __name__ == "__main__":
    main()
