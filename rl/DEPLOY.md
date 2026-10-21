# Running Fortress bots locally or on AWS

Everything ships as one Docker image (`rl/docker/Dockerfile`): the dedicated server with the RL
bridge, the Python trainer, and the trained models in `rl/models/`. The same image trains,
evaluates and serves.

| Model | File | Trained for | Last measured |
|---|---|---|---|
| 1 vs 1 | `models/fort_1v1.pt` | 1 vs 1 Fortress | 94% wins vs built-in AI, 99% vs an earlier version of itself |
| 6 vs 6 | `models/fort_6v6.pt` | 6 vs 6 Fortress | see the note stored in the file (`meta`) and `rl/README.md` |

## Build the image

From the repository root:

```bash
docker build -f rl/docker/Dockerfile -t fortbots .
```

The build compiles the server (a few minutes) and installs the CPU build of PyTorch. No GPU is
needed: the networks are small and the game simulation is the bottleneck.

## Play against the bots / watch them on your own computer

```bash
docker run --rm -p 4534:4534/udp fortbots
```

Then start Armagetron Advanced: Play Game > Multiplayer > Custom Connect > `localhost` port
`4534`. Join as a spectator to watch, or join a team: the bots fill whatever slots people leave
free. Without Docker, after building as in `rl/README.md`:

```bash
cd rl && ~/rlvenv/bin/python -m fortrl.serve --policy models/fort_6v6.pt --game configs/fort_6v6.cfg
```

## Public servers on AWS

1. **Instance.** Ubuntu 24.04, `t3.medium` is enough for three bot servers (each uses a fraction
   of a core in real time). 20 GB disk.
2. **Network.** Security group inbound rules: UDP 4534-4536 from `0.0.0.0/0` (game), TCP 22 from
   your own IP (SSH). Give the instance an Elastic IP so the address stays stable.
3. **Start.** Paste `rl/deploy/ec2-user-data.sh` as user data when launching (edit `REPO` and
   `BRANCH` first; a private repository needs a read-only token in the URL). It installs Docker,
   clones the repository and runs `rl/deploy/docker-compose.yml`: two 6v6 servers and one 1v1
   server, all announced to the master servers so they appear in the game's server browser
   within a few minutes.
4. **Operate.**
   ```bash
   cd /opt/armagetronad
   docker compose -f rl/deploy/docker-compose.yml ps        # status
   docker compose -f rl/deploy/docker-compose.yml logs -f   # round results, restarts
   git pull && docker compose -f rl/deploy/docker-compose.yml up -d --build   # new models/code
   ```

To change names, ports, models or rules edit `docker-compose.yml`; any game setting can be added
with `--set KEY=VALUE` (for example `--set SERVER_DNS=fort.example.com` if you point a domain at
the Elastic IP).

## Training on AWS

A compute optimized instance (for example `c7i.2xlarge`, 8 vCPU) runs one training job with three
to six game servers; the Python process uses about one core, each game server a fraction of one.
Keep runs on a volume so they survive the container:

```bash
mkdir -p ~/runs
docker run -d --name train6v6 -v ~/runs:/opt/fortrl/rl/runs fortbots \
    fortrl.train configs/train_6v6.toml --run-dir runs/m4_6v6 --init models/fort_1v1.pt
docker logs -f train6v6
# after a stop or a new instance: the same command with --resume instead of --init
```

Paths in the configs are relative to `rl/`; configs that refer to earlier runs (for example the
evaluation opponents in `train_6v6.toml`) expect those files under `runs/`. Copy results back
with `scp`, or sync `~/runs` to S3 (`aws s3 sync ~/runs s3://<bucket>/runs`). Turn a checkpoint
into a model for serving with:

```bash
docker run --rm -v ~/runs:/opt/fortrl/rl/runs -v $PWD/rl/models:/opt/fortrl/rl/models fortbots \
    fortrl.export runs/m4_6v6/checkpoints/latest.pt models/fort_6v6.pt --note "6v6 update N"
```

## Notes

- `SIMULATION_FIXED_STEP` (fast simulation) is for training only; `fortrl.serve` always runs in
  real time. Never train on a server people connect to.
- Each bot server keeps one Python process in lockstep with the game. If either dies, `serve`
  restarts both after five seconds, and Docker restarts the container if `serve` itself exits.
- Clients older than 0.2.8 may refuse the Fortress map; current clients connect normally.
