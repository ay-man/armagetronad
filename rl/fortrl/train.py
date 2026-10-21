"""PPO training of a Fortress agent.

    python -m fortrl.train configs/train_1v1_builtin.toml --run-dir runs/first
    python -m fortrl.train configs/train_1v1_builtin.toml --run-dir runs/first --resume
"""

import argparse
import json
import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .arena import Runner
from .evaluation import evaluate
from .factory import SnapshotPool, load_config, make_arenas, obs_spec, resolve
from .policy import PolicyBundle, PolicyController
from .ppo import PPOConfig, ppo_update
from .rewards import RewardConfig
from .stats import by_opponent, format_summary

log = logging.getLogger("fortrl.train")


def save_checkpoint(path, bundle, optimizer, state, cfg):
    tmp = path.with_suffix(".tmp")
    torch.save({"policy": bundle.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
                "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state()}}, tmp)
    tmp.replace(path)


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="continue from run-dir/checkpoints/latest.pt")
    parser.add_argument("--init", help="start a new run from the policy of this checkpoint")
    parser.add_argument("--total-steps", type=int, help="override run.total_steps")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    latest = ckpt_dir / "latest.pt"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        parser.error(f"{run_dir} is not empty; pass --resume to continue it or pick a new --run-dir")
    if args.resume and not latest.exists():
        parser.error(f"--resume given but {latest} does not exist")
    if args.resume and args.init:
        parser.error("--init only applies to new runs")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")])

    cfg = load_config(args.config)
    run_cfg = cfg.get("run", {})
    total_steps = args.total_steps or run_cfg.get("total_steps", 1_000_000)
    ppo_cfg = PPOConfig.from_dict(cfg.get("ppo", {}))
    reward_cfg = RewardConfig.from_dict(cfg.get("reward", {}))
    eval_cfg = cfg.get("eval", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    spec = obs_spec(cfg)
    if not args.resume:
        shutil.copy(cfg["_path"], run_dir / "config.toml")

    seed = run_cfg.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(run_cfg.get("torch_threads", 1))

    bundle = PolicyBundle(spec, hidden=ppo_cfg.hidden)
    optimizer = torch.optim.Adam(bundle.model.parameters(), lr=ppo_cfg.lr, eps=1e-5)
    state = {"update": 0, "steps": 0, "rounds": 0, "wall": 0.0}
    if args.init:
        bundle = PolicyBundle.load(args.init)
        if bundle.spec != spec:
            log.info("transferring --init policy from %s to %s", bundle.spec, spec)
            bundle = PolicyBundle.transfer(bundle, spec)
        optimizer = torch.optim.Adam(bundle.model.parameters(), lr=ppo_cfg.lr, eps=1e-5)
        log.info("initialized policy from %s", args.init)
    if args.resume:
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        bundle = PolicyBundle.from_state_dict(ckpt["policy"])
        optimizer = torch.optim.Adam(bundle.model.parameters(), lr=ppo_cfg.lr, eps=1e-5)
        optimizer.load_state_dict(ckpt["optimizer"])
        state = ckpt["state"]
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        log.info("resumed %s at update %d, %d steps", latest, state["update"], state["steps"])

    learner = PolicyController(bundle, "learner", learns=True)
    pool = SnapshotPool(run_dir / "pool", max_size=ckpt_cfg.get("pool_size", 20))
    if not args.resume:
        for path in ckpt_cfg.get("pool_seed", []):
            pool.add(PolicyBundle.load(resolve(path)), f"seed_{Path(path).parent.parent.name}_{Path(path).stem}")
        if ckpt_cfg.get("pool_seed"):
            log.info("seeded opponent pool with %d snapshots", len(ckpt_cfg["pool_seed"]))
    base_port = cfg.get("game", {}).get("base_port", 4800)
    arenas = make_arenas(cfg, learner, pool, spec, base_port, seed + state["update"])
    runner = Runner(arenas, spec, reward_cfg)
    log.info("training on %d arenas, obs size %d, until %d steps", len(arenas), spec.size, total_steps)

    eval_every = eval_cfg.get("every_updates", 10)
    try:
        while state["steps"] < total_steps:
            t0 = time.time()
            runner.run(lambda r: r.learner_steps >= ppo_cfg.rollout_steps)
            t_collect = time.time() - t0
            batch = runner.take_batch(ppo_cfg.gamma, ppo_cfg.lam)
            stats = ppo_update(bundle, optimizer, batch, ppo_cfg)
            bundle.norm.update(batch["obs"])
            records, runner.records = runner.records, []
            mine = [r for r in records if r["controller"] == "learner"]

            n = len(batch["obs"])
            state["update"] += 1
            state["steps"] += n
            state["rounds"] += len(mine)
            elapsed = time.time() - t0
            state["wall"] += elapsed
            row = {"update": state["update"], "steps": state["steps"], "rounds": state["rounds"],
                   "sps": n / elapsed, "collect_s": t_collect, "update_s": elapsed - t_collect,
                   **stats, "train": by_opponent(mine)}
            append_jsonl(run_dir / "metrics.jsonl", row)
            log.info("update %d  steps %d  %.0f steps/s  kl %.4f  ent %.3f  ev %.2f", state["update"],
                     state["steps"], row["sps"], stats["approx_kl"], stats["entropy"], stats["explained_variance"])
            for opp, s in row["train"].items():
                log.info("  train %s", format_summary(opp, s))

            if state["update"] % ckpt_cfg.get("every_updates", 10) == 0:
                save_checkpoint(ckpt_dir / f"ckpt_{state['update']:06d}.pt", bundle, optimizer, state, cfg)
                save_checkpoint(latest, bundle, optimizer, state, cfg)
            if ckpt_cfg.get("pool_every_updates") and state["update"] % ckpt_cfg["pool_every_updates"] == 0:
                pool.add(bundle, f"u{state['update']:06d}")
            if eval_every and state["update"] % eval_every == 0:
                run_eval(cfg, bundle, spec, state, run_dir, eval_cfg, base_port + 100)
    finally:
        save_checkpoint(latest, bundle, optimizer, state, cfg)
        runner.close()
        log.info("saved %s", latest)


def run_eval(cfg, bundle, spec, state, run_dir, eval_cfg, port):
    controller = PolicyController(bundle, "eval", greedy=eval_cfg.get("greedy", False))
    t0 = time.time()
    opponents = []
    for opp in eval_cfg.get("opponents", ["builtin", "random"]):
        if opp.startswith("ckpt:") and not Path(opp[len("ckpt:"):]).exists():
            log.warning("  skipping evaluation opponent %s: file not found", opp)
            continue
        opponents.append(opp)
    results = evaluate(cfg, controller, opponents,
                       eval_cfg.get("rounds", 100), spec, port, seed=state["update"])
    append_jsonl(run_dir / "eval.jsonl", {"update": state["update"], "steps": state["steps"], "results": results})
    for opp, s in results.items():
        log.info("  EVAL %s", format_summary(opp, s))
    log.info("  eval took %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
