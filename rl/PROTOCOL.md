# RL bridge protocol (version 1)

The dedicated server connects to a Unix domain socket named by `RL_BRIDGE_SOCKET`
and then runs in lockstep with the trainer: every `RL_DECISION_INTERVAL` seconds of
game time it sends one message and blocks until the reply arrives. No game time
passes while the trainer thinks.

Implementation: `src/tron/gRLBridge.cpp` (server), `rl/fortrl/server.py` (trainer).

## Server settings

| Setting | Default | Meaning |
|---|---|---|
| `RL_BRIDGE_SOCKET` | empty (off) | socket path to connect to |
| `RL_BRIDGE_TEAMS` | 1 | AI players on teams with id `< n` are steered by the trainer; the rest keep the built-in AI. Negative: every AI player is steered. 0: nothing is steered, the trainer only observes |
| `RL_DECISION_INTERVAL` | 0.1 | game seconds between decisions |
| `RL_RAYS` | 16 | wall sensor rays per steered cycle, evenly spread, ray 0 straight ahead, counter-clockwise |
| `RL_RAY_RANGE` | 250 | sensor range |
| `SIMULATION_FIXED_STEP` | 0 (real time) | if > 0, each frame advances the clock by this many seconds regardless of real time |
| `DEDICATED_AI_ONLY_GAMES` | 0 | play rounds even when no client is connected |

## Server → trainer

One JSON object per line.

```json
{"type": "step", "version": 1, "round": 3, "time": 4.2, "winner": 0,
 "decision_interval": 0.1, "ray_range": 250,
 "agents": [{"id": 20, "team": 0, "name": "gltron", "alive": 1,
             "x": 245, "y": 450, "dx": 0, "dy": -1, "speed": 20, "braking": 0,
             "rubber": 0, "brake_reservoir": 1, "can_turn_left": 1, "can_turn_right": 1,
             "rays": [[250, 0], [130.6, 1], "..."]}],
 "cycles": [{"id": 19, "team": 1, "name": "armagetronad", "alive": 1,
             "x": 255, "y": 50, "dx": 0, "dy": 1, "speed": 20, "braking": 0}],
 "teams": [{"id": 0, "score": 0, "alive": 1, "name": "gltron"}],
 "zones": [{"team": 1, "x": 250, "y": 50, "radius": 40, "conquered": 0, "safe": 1}],
 "events": ["DEATH_FRAG armagetronad gltron"]}
```

- `type`: `step` (reply with actions) or `round_end` (reply `OK`). `round_end` is sent once
  the round winner is decided; the server keeps simulating the end-of-round phase without
  further messages until the next round starts.
- `round`: counts rounds since the server started; a new value means a new episode.
- `winner`: 0 undecided, -1 draw/no legitimate winner, `n > 0` team `n-1` won.
- `agents`: steered cycles with their senses. `rays[i] = [distance, wall type]`, wall
  type 0 nothing, 1 arena rim, 2 enemy wall, 3 teammate wall, 4 own wall.
- `cycles`: every cycle in the game (positions are absolute, `dx/dy` is the unit heading).
- `zones`: fortress zones; `conquered` runs from 0 to 1, `safe` turns 0 once conquered.
- `events`: ladderlog lines since the previous message (see the game's ladderlog
  documentation), e.g. `DEATH_FRAG prey hunter`, `BASEZONE_CONQUERED team x y`,
  `ROUND_WINNER team players`.

## Trainer → server

One line per message:

- `A <id> <turn> <brake> [<id> <turn> <brake> ...]` — `turn` is -1 (left), 0, 1 (right);
  `brake` 0 or 1 is held until the next decision. Agents left out keep their brake state and do not turn.
- `OK` — acknowledges `round_end`.
- `QUIT` — the server shuts down.

If the socket closes or a malformed line arrives, the server logs the reason and exits.
