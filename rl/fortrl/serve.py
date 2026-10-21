"""Runs a public, real-time game server where a trained model drives every AI cycle.

Players connect with the normal Armagetron Advanced client to spectate or to join a team
(the bots fill whatever team slots humans leave free):

    python -m fortrl.serve --game configs/fort_6v6.cfg --policy models/fort_6v6.pt \\
        --name "Fort Bots 6v6" --port 4534 --public

--public announces the server to the master servers so it shows up in the in-game
server browser; the UDP port must then be reachable from the internet. Without it,
connect directly: Play Game > Multiplayer > Custom Connect > <host>:<port>.
The server restarts by itself if it or the model process fails.
"""

import argparse
import logging
import time
from pathlib import Path

from .arena import Arena, Runner
from .factory import resolve
from .policy import PolicyBundle, PolicyController
from .rewards import RewardConfig
from .server import GameServer, ServerError, default_binary
from .stats import summarize

log = logging.getLogger("fortrl.serve")


class _Everyone(dict):
    """Controller assignment that hands every team to the same model."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def get(self, team, default=None):
        return self.controller


def parse_settings(items):
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        out[key.strip().upper()] = value.strip()
    return out


def serve_once(args, bundle):
    controller = PolicyController(bundle, "policy", greedy=args.greedy)
    settings = {
        "RL_RAYS": bundle.spec.rays,
        "SIMULATION_FIXED_STEP": 0,          # real time
        "SERVER_NAME": args.name,
        "TALK_TO_MASTER": 1 if args.public else 0,
        "MAX_CLIENTS": args.max_clients,
        "TEAM_BALANCE_WITH_AIS": 1,
        "SP_TEAM_BALANCE_WITH_AIS": 1,
    }
    settings.update(parse_settings(args.set))
    workdir = Path(args.workdir) if args.workdir else None
    if workdir:
        workdir.mkdir(parents=True, exist_ok=True)
    server = GameServer(args.server or default_binary(), resolve(args.game), steered_teams=-1, port=args.port,
                        extra_settings=settings, workdir=workdir, connect_timeout=120).start()
    log.info("serving %r on UDP port %d (log: %s)", args.name, args.port, server.log_path)
    arena = Arena(server, "serve", lambda r: _Everyone(controller))
    runner = Runner([arena], bundle.spec, RewardConfig())
    try:
        while True:
            runner.run(lambda r: len({(x["round"], x["team"]) for x in r.records}) >= 2 * args.report_rounds)
            results = {}
            for x in runner.records:
                results.setdefault(x["team"], []).append(x)
            log.info("last %d rounds: %s", args.report_rounds, "  ".join(
                f"team {t + 1} won {summarize(rs)['win_rate']:.0%}" for t, rs in sorted(results.items())))
            runner.records = []
    finally:
        runner.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="trained model (.pt)")
    parser.add_argument("--game", default="configs/fort_6v6.cfg", help="game rules file")
    parser.add_argument("--name", default="Fort Bots", help="server name shown in the server browser")
    parser.add_argument("--port", type=int, default=4534, help="UDP port players connect to")
    parser.add_argument("--public", action="store_true", help="announce to the master servers")
    parser.add_argument("--max-clients", type=int, default=16)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="extra game setting, repeatable (e.g. --set SERVER_DNS=fort.example.com)")
    parser.add_argument("--server", help="dedicated server binary (default: FORTRL_SERVER or the build tree)")
    parser.add_argument("--workdir", help="keep config, logs and ladderlog here instead of a temp dir")
    parser.add_argument("--greedy", action="store_true", help="always take the model's top choice")
    parser.add_argument("--report-rounds", type=int, default=10)
    parser.add_argument("--no-restart", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bundle = PolicyBundle.load(args.policy)
    while True:
        try:
            serve_once(args, bundle)
        except KeyboardInterrupt:
            return
        except (ServerError, TimeoutError, OSError) as e:
            log.error("server stopped: %s", e)
            if args.no_restart:
                raise SystemExit(1)
            time.sleep(5)
            log.info("restarting")


if __name__ == "__main__":
    main()
