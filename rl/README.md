# fortrl: self-learning Fortress bots for Armagetron Advanced

Bots that learn to play Fortress by playing it, using reinforcement learning (PPO).
Nothing about strategy is programmed in: a bot only sees what its cycle senses, can
turn left, right or brake, and is rewarded for winning. Everything else comes from
experience.

## How it fits together

```
 Python trainer (rl/fortrl)                     armagetronad-dedicated (C++)
 ┌───────────────────────────┐   unix socket    ┌──────────────────────────────────┐
 │ policy network (PyTorch)  │◀── observations ─│ gRLBridge: every 0.1 game seconds │
 │ PPO, opponent schedule,   │─── actions ─────▶│ sends state, waits, applies turns │
 │ evaluation, checkpoints   │                  │ fixed-step clock: no real-time    │
 └───────────────────────────┘                  │ waiting, ~270x faster than real   │
        ▲  one trainer drives several servers   └──────────────────────────────────┘
```

- The game is the real game: same physics, same Fortress rules, same built-in AI.
- The server runs in lockstep with the trainer, so learning never misses a frame and
  runs as fast as the CPU allows (about 2000 decisions/second per server).
- A server steers AI players on teams `0 .. RL_BRIDGE_TEAMS-1` from Python. The other
  teams keep the game's own AI, which serves as a fixed outside opponent.
- Wire format: [PROTOCOL.md](PROTOCOL.md).

### Fortress, as implemented in `src/tron/zone/zFortress.cpp`

Every team owns a fortress zone (the one nearest its spawn). Enemies inside a zone raise
its conquest level, defenders inside lower it, and it decays on its own:
`d(conquered)/dt = CONQUEST_RATE * enemies - DEFEND_RATE * defenders - DECAY_RATE`.
At 1 the zone is conquered and collapses. With the soccer rules used here
(`config/examples/fortress_soccer.cfg`) a lone attacker needs a few seconds, one defender
inside stops it, and the round is won by the last team with a standing fortress or the
last team alive. Cycles die by hitting walls, as always.

### What a bot observes (`fortrl/features.py`, 111 numbers for 1 vs 1)

Everything is relative to its own position and heading:

- 16 wall sensor rays around it: distance and what was hit (rim, enemy wall, own wall)
- own speed, rubber, brake reserve, whether it can turn right now, round time
- nearest enemy: relative position, distance, heading, speed, alive
- own and enemy fortress: relative position, distance, radius, conquest level, whether it is inside

### Actions

Six choices every 0.1 s: straight, left or right, each with or without brake.

### Rewards (`fortrl/rewards.py`, weights in the config)

| term | default | why |
|---|---|---|
| win / loss / draw | +1 / -1 / 0 | the actual objective |
| enemy fortress conquest progress | +1 per full conquest | makes attacking discoverable |
| own fortress conquest progress | -1 per full conquest | makes defending discoverable |
| kill | +0.5 | |
| death | -0.5 | |
| alive per second | 0 | off: rewarding survival alone breeds passive bots |

Each term is logged separately per round so you can see what the bot is being paid for.

### Learning

PPO with GAE, a two-layer 256 unit actor and critic, running observation normalization.
PPO was chosen because it is stable and simple to tune with discrete actions and many cheap
parallel episodes, which is exactly what the fast simulation gives us. No recurrent
memory and no league yet: get a single bot learning first.

## Setup

Tested on Ubuntu 24.04.

```bash
# build dependencies
sudo apt-get install -y autoconf automake libtool bison make g++ pkg-config \
    libxml2-dev libboost-dev libboost-thread-dev libboost-system-dev \
    libprotobuf-dev protobuf-compiler libcurl4-openssl-dev python3-venv

# dedicated server build used for training (optimized, no graphics)
./bootstrap.sh
mkdir -p build/rl_server && cd build/rl_server
../../configure --enable-dedicated --disable-glout --disable-sysinstall --disable-desktop \
    --disable-etc --disable-useradd --disable-master CXXFLAGS="-O2 -g"
make -j4 && make armagetronad-dedicated
cd ../..

# Python environment
python3 -m venv ~/rlvenv
~/rlvenv/bin/pip install -r rl/requirements.txt
```

The trainer looks for `build/rl_server/armagetronad-dedicated`; set `FORTRL_SERVER` or
`[game] server` in the config to use another binary.

## Verify

```bash
cd rl
~/rlvenv/bin/python -m pytest                     # Python unit tests
cd .. && batch/test_builds.sh client_debug      # game build + C++ unit tests
```

## Run

All commands from the `rl/` directory.

```bash
# baseline: how does random play do?
~/rlvenv/bin/python -m fortrl.evaluate configs/train_1v1_builtin.toml --policy random --rounds 300

# milestone 1: learn against the built-in AI
~/rlvenv/bin/python -m fortrl.train configs/train_1v1_builtin.toml --run-dir runs/m1_builtin

# continue after stopping (Ctrl-C saves runs/m1_builtin/checkpoints/latest.pt)
~/rlvenv/bin/python -m fortrl.train configs/train_1v1_builtin.toml --run-dir runs/m1_builtin --resume

# evaluate any checkpoint in fresh matches
~/rlvenv/bin/python -m fortrl.evaluate configs/train_1v1_builtin.toml \
    --policy runs/m1_builtin/checkpoints/latest.pt --opponents builtin random --rounds 500

# compare against an earlier version of itself
~/rlvenv/bin/python -m fortrl.evaluate configs/train_1v1_builtin.toml \
    --policy runs/m1_builtin/checkpoints/latest.pt \
    --opponents ckpt:runs/m1_builtin/checkpoints/ckpt_000050.pt --rounds 500

# curves: writes runs/m1_builtin/progress.png
~/rlvenv/bin/python -m fortrl.plot runs/m1_builtin

# milestone 3: self-play league, starting from the milestone 1 bot
~/rlvenv/bin/python -m fortrl.train configs/train_1v1_selfplay.toml --run-dir runs/m3_league \
    --init runs/m1b_gamma997/checkpoints/ckpt_000360.pt

# milestone 4: 6 vs 6 teams, starting from the shipped 1 vs 1 model. Its weights are carried
# over to the bigger observation (3 nearest teammates and enemies) automatically.
~/rlvenv/bin/python -m fortrl.train configs/train_6v6.toml --run-dir runs/m4_6v6 --init models/fort_1v1.pt

# turn a checkpoint into a small model file for serving or as a starting point
~/rlvenv/bin/python -m fortrl.export runs/m4_6v6/checkpoints/latest.pt models/fort_6v6.pt --note "6v6 update N"
```

Trained models ship in `models/` (`fort_1v1.pt`, `fort_6v6.pt`); each file stores a note on where
it came from. Running them locally, in Docker or on AWS: [DEPLOY.md](DEPLOY.md).

### Watch the bots

```bash
# record matches at several training stages and build a replay page you open in a browser
~/rlvenv/bin/python -m fortrl.record configs/train_1v1_builtin.toml --rounds 3 --out runs/replays.json \
    --policy random --label "Random play" \
    --policy runs/m1_builtin/checkpoints/ckpt_000010.pt --label "Early" \
    --policy runs/m1_builtin/checkpoints/latest.pt --label "Latest"
~/rlvenv/bin/python -m fortrl.replay_page runs/replays.json --out runs/replays.html \
    --stage "Random play=runs/baseline_random.json" --stage "Latest=runs/m1_builtin:100"

# or watch live with the real game: a real-time server where the model drives every AI cycle;
# connect with a normal game client (Multiplayer, Custom Connect, <host>:4534) to spectate or play.
# --public lists it in the in-game server browser (see DEPLOY.md for running it on AWS)
~/rlvenv/bin/python -m fortrl.serve --policy models/fort_6v6.pt --game configs/fort_6v6.cfg --name "My Fort Bots"
```

### What a run directory holds

| file | content |
|---|---|
| `config.toml` | the config the run was started with |
| `train.log` | human readable progress |
| `metrics.jsonl` | one line per PPO update: speed, losses, entropy, per opponent training results |
| `eval.jsonl` | periodic evaluation in separate matches against fixed opponents |
| `checkpoints/` | `ckpt_<update>.pt` every N updates, `latest.pt` always |
| `pool/` | frozen past versions used as self-play opponents |

## Reading the results

Only `eval.jsonl` (and `fortrl.evaluate`) count as evidence: those matches use fresh
servers, no learning, and report a 95% confidence interval for the win rate. Training
numbers are measured against opponents the bot is adapting to and are optimistic.
Learning is working when evaluation win rate against the built-in AI and against earlier
checkpoints rises well above the random baseline's interval.

## Known limitations

- Team arenas assign one controller per team: all six cycles of a team share one model, and
  league opponents are whole teams (self, pool snapshot or random).
- Against the built-in AI the learner always drives team 0 (`RL_BRIDGE_TEAMS` steers the
  lowest team ids), so it always starts from the same spawn on a given map.
- One map (`for_old_clients`). Generalization across maps is not measured yet.
- The game's own randomness is not seeded; runs are statistically, not bit-for-bit, reproducible.
- `SIMULATION_FIXED_STEP` also speeds up real-time network timeouts, so never use it on a
  server people connect to.
