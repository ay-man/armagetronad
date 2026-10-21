"""Drives one or more game servers in lockstep and collects experience.

Each server ("arena") has a fixed set of steered teams. At the start of
every round an opponent function decides which controller drives each
steered team; the remaining teams use the game's built-in AI.
"""

from dataclasses import dataclass, field
import logging
import selectors

import numpy as np

from .features import ACTIONS, encode
from .rewards import RewardBreakdown, outcome, outcome_reward, parse_events, step_reward

log = logging.getLogger(__name__)


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    logp: float
    value: float
    reward: float = 0.0
    done: bool = False


@dataclass
class AgentState:
    pid: int
    team: int
    name: str
    controller: object
    prev_msg: dict = None
    pending: Transition = None
    pending_reward: float = 0.0
    finished: list = field(default_factory=list)  # finalized transitions not yet consumed
    breakdown: RewardBreakdown = field(default_factory=RewardBreakdown)
    alive: bool = True
    steps: int = 0
    start_time: float = 0.0
    death_time: float = None
    kills: int = 0

    def finalize(self, done):
        t = self.pending
        t.reward, t.done = self.pending_reward, done
        self.pending, self.pending_reward = None, 0.0
        if self.controller.learns:
            self.finished.append(t)


class Arena:
    def __init__(self, server, name, opponent_fn):
        """opponent_fn(round) -> {team_id: Controller} for every steered team."""
        self.server = server
        self.name = name
        self.opponent_fn = opponent_fn
        self.round = None
        self.assignment = {}
        self.agents = {}
        self.num_teams = 0


class Runner:
    def __init__(self, arenas, spec, reward_cfg):
        self.arenas = arenas
        self.spec = spec
        self.reward_cfg = reward_cfg
        self.records = []           # one dict per finished agent-round
        self.open_trajectories = {}  # (arena idx, round, pid) -> AgentState, learner agents only
        self.closed = []            # lists of transitions of finished learner rounds
        self.learner_steps = 0      # finalized learner transitions not yet consumed
        self.total_steps = 0        # all decisions taken, any controller
        self.observers = []         # callables (arena idx, msg, actions or None), e.g. replay recording
        self.selector = selectors.DefaultSelector()
        for i, arena in enumerate(arenas):
            self.selector.register(arena.server.conn, selectors.EVENT_READ, i)

    # ------------------------------------------------------------------ main loop

    def run(self, until):
        """Processes messages until until(self) is true.

        All servers that are waiting at the same time get their decisions in one batch.
        """
        while not until(self):
            ready = [i for i, a in enumerate(self.arenas) if a.server.has_buffered()]
            if not ready:
                ready = [key.data for key, _ in self.selector.select(timeout=60)]
                if not ready:
                    raise TimeoutError("no game server sent anything for 60s")
            acting = []
            for i in ready:
                acting += self._handle(i, self.arenas[i].server.receive())
            self._act(acting)

    def _handle(self, idx, msg):
        """Bookkeeping for one message; returns the agents that need a decision."""
        arena = self.arenas[idx]
        arena.num_teams = len(msg["teams"])
        if msg["round"] != arena.round:
            self._abort_round(idx, arena)
            arena.round = msg["round"]
            arena.assignment = arena.opponent_fn(arena.round)
        for observe in self.observers:
            observe(idx, msg, None)
        if msg["type"] == "round_end":
            self._end_round(idx, arena, msg)
            arena.server.send_ok()
            return []
        acting = self._step(idx, arena, msg)
        if not acting:
            arena.server.send_actions({})
        return acting

    def _step(self, idx, arena, msg):
        acting = []
        for a in msg["agents"]:
            alive = bool(a["alive"]) and "x" in a
            st = arena.agents.get(a["id"])
            if st is None:
                if not alive:
                    continue
                controller = arena.assignment.get(a["team"])
                if controller is None:
                    raise RuntimeError(f"arena {arena.name}: no controller for steered team {a['team']}")
                st = AgentState(a["id"], a["team"], a["name"], controller, start_time=msg["time"])
                arena.agents[a["id"]] = st
                if controller.learns:
                    self.open_trajectories[(idx, arena.round, a["id"])] = st
            elif not st.alive:
                continue
            else:
                self._accumulate(st, msg, alive)
                if not alive:
                    st.alive, st.death_time = False, msg["time"]
                    continue
                st.finalize(done=False)
                self._count(st)
            st.prev_msg = msg
            acting.append((idx, st, encode(msg, a, self.spec)))
        return acting

    def _act(self, acting):
        """Batches decisions per controller across arenas and sends them."""
        by_controller = {}
        for item in acting:
            by_controller.setdefault(id(item[1].controller), []).append(item)
        replies = {idx: {} for idx, _, _ in acting}
        for group in by_controller.values():
            act, logp, value = group[0][1].controller.act(np.stack([o for _, _, o in group]))
            for (idx, st, o), a_, lp, v in zip(group, act, logp, value):
                st.pending = Transition(o, int(a_), float(lp), float(v))
                st.steps += 1
                replies[idx][st.pid] = ACTIONS[int(a_)]
        self.total_steps += len(acting)
        for idx, actions in replies.items():
            for observe in self.observers:
                observe(idx, None, actions)
            self.arenas[idx].server.send_actions(actions)

    def _accumulate(self, st, msg, now_alive):
        part = step_reward(self.reward_cfg, st.prev_msg, msg, {"team": st.team, "name": st.name},
                           st.alive, now_alive)
        st.pending_reward += part.total
        for k, v in part.parts.items():
            st.breakdown.add(k, v)
        st.kills += sum(1 for kind, f in parse_events(msg["events"])
                        if kind == "DEATH_FRAG" and len(f) >= 2 and f[1] == st.name and f[0] != st.name)
        st.prev_msg = msg

    def _count(self, st):
        if st.controller.learns:
            self.learner_steps += 1

    def _end_round(self, idx, arena, msg):
        ended = {a["id"]: a for a in msg["agents"]}
        for st in arena.agents.values():
            if st.pending is None:
                continue
            if st.alive:
                still = ended.get(st.pid, {}).get("alive", 0)
                self._accumulate(st, msg, bool(still))
                if not still:
                    st.alive, st.death_time = False, msg["time"]
            result = outcome(msg, st.team)
            part = outcome_reward(self.reward_cfg, result)
            st.pending_reward += part.total
            st.breakdown.add("outcome", part.total)
            st.finalize(done=True)
            self._count(st)
            self._close(idx, arena, st, result, msg["time"])
        arena.agents.clear()

    def _abort_round(self, idx, arena):
        """A round ended without a round_end message: cut trajectories without outcome."""
        for st in arena.agents.values():
            if st.pending is not None:
                st.finalize(done=True)
                self._count(st)
                self._close(idx, arena, st, "aborted", None)
        if arena.agents:
            log.warning("arena %s: round %s ended without result", arena.name, arena.round)
        arena.agents.clear()

    def _close(self, idx, arena, st, result, end_time):
        if st.controller.learns:
            self.open_trajectories.pop((idx, arena.round, st.pid), None)
            self.closed.append(st.finished)
            st.finished = []
        opponents = sorted({c.name for t, c in arena.assignment.items() if t != st.team})
        if 0 <= arena.server.steered_teams < arena.num_teams:
            opponents.append("builtin")
        end = st.death_time if st.death_time is not None else end_time
        self.records.append({
            "arena": arena.name, "round": arena.round, "team": st.team, "controller": st.controller.name,
            "opponent": "+".join(opponents), "result": result, "reward": st.breakdown.total,
            "parts": dict(st.breakdown.parts), "steps": st.steps, "kills": st.kills,
            "died": st.death_time is not None,
            "survival": (end - st.start_time) if end is not None else None,
        })

    # ------------------------------------------------------------------ learner data

    def take_batch(self, gamma, lam):
        """GAE over all finalized learner transitions; consumes them."""
        segments = [(traj, 0.0) for traj in self.closed if traj]
        for st in self.open_trajectories.values():
            if st.finished:
                bootstrap = st.pending.value if st.pending is not None else 0.0
                segments.append((st.finished, bootstrap))
                st.finished = []
        self.closed = []
        self.learner_steps = 0

        obs, act, logp, val, adv, ret = [], [], [], [], [], []
        for traj, bootstrap in segments:
            gae = 0.0
            advs = np.zeros(len(traj), dtype=np.float32)
            for t in reversed(range(len(traj))):
                tr = traj[t]
                next_value = bootstrap if t == len(traj) - 1 else traj[t + 1].value
                if tr.done:
                    next_value, gae = 0.0, 0.0
                delta = tr.reward + gamma * next_value - tr.value
                gae = delta + gamma * lam * gae
                advs[t] = gae
            for tr, a in zip(traj, advs):
                obs.append(tr.obs)
                act.append(tr.action)
                logp.append(tr.logp)
                val.append(tr.value)
                adv.append(a)
                ret.append(a + tr.value)
        if not obs:
            return None
        return {
            "obs": np.stack(obs), "act": np.asarray(act), "logp": np.asarray(logp, dtype=np.float32),
            "val": np.asarray(val, dtype=np.float32), "adv": np.asarray(adv, dtype=np.float32),
            "ret": np.asarray(ret, dtype=np.float32),
        }

    def close(self):
        self.selector.close()
        for arena in self.arenas:
            arena.server.close()
