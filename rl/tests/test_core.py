import math

import numpy as np
import pytest

from fortrl.arena import AgentState, Runner, Transition
from fortrl.features import ObsSpec, RunningNorm, encode
from fortrl.rewards import RewardConfig, conquest_delta, outcome, step_reward


def make_msg(agent_dir=(0, 1), enemy=(100, 200), zones=None, time=0.0, events=(), winner=0):
    agent = {"id": 1, "team": 0, "name": "me", "alive": 1, "x": 100.0, "y": 100.0,
             "dx": agent_dir[0], "dy": agent_dir[1], "speed": 20.0, "braking": 0, "rubber": 0.0,
             "brake_reservoir": 1.0, "can_turn_left": 1, "can_turn_right": 1,
             "rays": [[50.0, 1]] + [[250.0, 0]] * 15}
    other = {"id": 2, "team": 1, "name": "you", "alive": 1, "x": float(enemy[0]), "y": float(enemy[1]),
             "dx": 0, "dy": -1, "speed": 20.0, "braking": 0}
    return {"type": "step", "round": 1, "time": time, "winner": winner, "ray_range": 250.0,
            "agents": [agent], "cycles": [agent, other], "teams": [], "events": list(events),
            "zones": zones if zones is not None else [
                {"team": 0, "x": 100.0, "y": 50.0, "radius": 40.0, "conquered": 0.0, "safe": 1},
                {"team": 1, "x": 100.0, "y": 450.0, "radius": 40.0, "conquered": 0.0, "safe": 1}]}


def test_encode_size_and_frame():
    spec = ObsSpec()
    msg = make_msg()
    obs = encode(msg, msg["agents"][0], spec)
    assert obs.shape == (spec.size,)
    assert obs[0] == pytest.approx(0.2)            # first ray: 50 / 250
    assert list(obs[1:5]) == [1, 0, 0, 0]          # rim
    enemy = obs[16 * 5 + 7:16 * 5 + 7 + 8]
    assert enemy[0] == 1 and enemy[2] == pytest.approx(100 / 500)  # straight ahead
    assert enemy[3] == pytest.approx(0)
    assert enemy[5] == pytest.approx(-1)            # heading towards us


def test_encode_is_rotation_invariant():
    spec = ObsSpec()
    a = make_msg(agent_dir=(0, 1), enemy=(100, 200), zones=[])
    b = make_msg(agent_dir=(1, 0), enemy=(200, 100), zones=[])
    b["cycles"][1]["dx"], b["cycles"][1]["dy"] = -1, 0
    assert np.allclose(encode(a, a["agents"][0], spec), encode(b, b["agents"][0], spec), atol=1e-6)


def test_encode_rejects_ray_mismatch():
    msg = make_msg()
    with pytest.raises(ValueError):
        encode(msg, msg["agents"][0], ObsSpec(rays=8))


def test_conquest_delta_counts_vanished_zone_as_conquered():
    prev = make_msg()
    prev["zones"][1]["conquered"] = 0.6
    cur = make_msg(zones=[prev["zones"][0]])
    enemy, own = conquest_delta(prev, cur, team=0)
    assert enemy == pytest.approx(0.4) and own == 0


def test_step_reward_parts():
    cfg = RewardConfig(kill=0.5, death=-0.5, alive=0.1)
    prev = make_msg(time=1.0)
    cur = make_msg(time=1.5, events=["DEATH_FRAG you me"])
    cur["zones"][0]["conquered"] = 0.25
    r = step_reward(cfg, prev, cur, {"team": 0, "name": "me"}, True, False)
    assert r.parts == pytest.approx({"own_conquest": -0.25, "alive": 0.05, "death": -0.5, "kill": 0.5})


def test_outcome():
    assert outcome({"winner": 1}, 0) == "win"
    assert outcome({"winner": 2}, 0) == "loss"
    assert outcome({"winner": -1}, 0) == "draw"
    assert outcome({"winner": 0}, 0) == "aborted"


def test_running_norm_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3, 2, size=(1000, 4))
    norm = RunningNorm(4)
    for chunk in np.split(data, 10):
        norm.update(chunk)
    assert np.allclose(norm.mean, data.mean(0), atol=1e-3)
    assert np.allclose(norm.var, data.var(0), atol=1e-2)


class _Learner:
    learns = True
    name = "learner"


def test_gae_matches_reference():
    runner = Runner.__new__(Runner)
    runner.closed, runner.open_trajectories, runner.learner_steps = [], {}, 0
    gamma, lam = 0.9, 0.8
    rewards, values = [1.0, 0.0, 2.0], [0.5, 0.4, 0.3]
    traj = [Transition(np.zeros(1), 0, 0.0, v, r, i == 2) for i, (r, v) in enumerate(zip(rewards, values))]
    runner.closed.append(traj)
    # an open trajectory, bootstrapped from its pending transition
    st = AgentState(1, 0, "me", _Learner())
    st.finished = [Transition(np.zeros(1), 0, 0.0, 0.2, 1.0, False)]
    st.pending = Transition(np.zeros(1), 0, 0.0, 0.7)
    runner.open_trajectories[(0, 1, 1)] = st

    batch = runner.take_batch(gamma, lam)
    expected = []
    gae = 0.0
    for t in reversed(range(3)):
        nv = 0.0 if t == 2 else values[t + 1]
        delta = rewards[t] + gamma * nv - values[t]
        gae = delta + gamma * lam * (gae if t < 2 else 0.0)
        expected.insert(0, gae)
    expected.append(1.0 + gamma * 0.7 - 0.2)
    assert np.allclose(batch["adv"], expected)
    assert np.allclose(batch["ret"], np.array(expected) + np.array(values + [0.2]))
    assert st.finished == [] and st.pending is not None


def test_wilson_bounds():
    from fortrl.stats import wilson
    lo, hi = wilson(0, 300)
    assert lo == 0 and hi < 0.02
    lo, hi = wilson(150, 300)
    assert lo < 0.5 < hi and math.isclose(lo + hi, 1.0, abs_tol=1e-9)
