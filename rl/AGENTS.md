# rl/ Directory

## Summary

Python reinforcement learning system that trains Fortress bots by playing the real game.
User documentation: `README.md`. Wire protocol: `PROTOCOL.md`.

## Details

The trainer starts `armagetronad-dedicated` processes with the C++ bridge enabled
(`src/tron/gRLBridge.cpp`) and drives them in lockstep over Unix sockets. Game rules live in
`.cfg` files fed to the server; learning settings live in `.toml` files read by the trainer.

## Directory Structure

```
.
├── README.md            # setup, commands, design rationale
├── PROTOCOL.md          # bridge message format
├── DEPLOY.md            # Docker, local and AWS deployment
├── requirements.txt     # Python dependencies
├── pytest.ini           # test configuration (run pytest from rl/)
├── configs/
│   ├── fort_1v1.cfg              # game rules for 1 vs 1 Fortress training
│   ├── train_1v1_builtin.toml    # milestone 1: learner vs built-in AI
│   ├── train_1v1_selfplay.toml   # milestone 3: self-play league
│   ├── train_1v1_selfplay_m3b.toml  # continuation with the rebalanced mix
│   ├── fort_6v6.cfg              # game rules for 6 vs 6 Fortress
│   ├── train_6v6.toml            # milestone 4: 6 vs 6 team training
│   └── watch_6v6.toml            # 6 vs 6 recording / evaluation
├── fortrl/
│   ├── server.py        # launches a server, speaks the protocol
│   ├── features.py      # egocentric observation encoding, action table, running normalizer
│   ├── rewards.py       # reward terms from consecutive messages
│   ├── arena.py         # multi-server lockstep runner, trajectory bookkeeping, GAE
│   ├── policy.py        # actor-critic network and controllers (learner, frozen, random)
│   ├── ppo.py           # PPO update
│   ├── factory.py       # config loading, arenas, opponent schedules, snapshot pool
│   ├── evaluation.py    # evaluation matches on fresh servers
│   ├── stats.py         # win rates with Wilson intervals, round summaries
│   ├── train.py         # CLI: train / resume
│   ├── evaluate.py      # CLI: evaluate a checkpoint or the random baseline
│   ├── record.py        # CLI: record matches for the replay viewer
│   ├── replay_page.py   # CLI: build the HTML replay/progress page (template replay_viewer.html)
│   ├── serve.py         # CLI: public real-time bot server
│   ├── export.py        # CLI: checkpoint -> small model file
│   └── plot.py          # CLI: progress plots
├── models/              # shipped trained models (policy only)
├── docker/              # Dockerfile (build from the repository root) and its context ignore list
├── deploy/              # docker-compose for several bot servers, EC2 user data
└── tests/               # pytest unit tests (no game server needed)
```

## Conventions

- Python 3.11+, standard library `tomllib`, PyTorch, NumPy.
- Runs go to `rl/runs/` (git ignored).
- Observation layout changes must update `ObsSpec.size` and `tests/test_core.py`; checkpoints
  store the `ObsSpec` they were trained with and refuse mismatching configs.
- Bridge protocol changes must bump the version in `gRLBridge.cpp` and `server.py` and update `PROTOCOL.md`.
