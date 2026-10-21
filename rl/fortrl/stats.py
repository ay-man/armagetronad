"""Aggregation of round records into the metrics we judge learning by."""

import math

import numpy as np


def wilson(successes, n, z=1.96):
    """95% Wilson score interval for a rate."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(records):
    """Win/draw/loss rates, survival, kills and reward parts over decided round records."""
    aborted = sum(r["result"] == "aborted" for r in records)
    records = [r for r in records if r["result"] != "aborted"]
    n = len(records)
    if n == 0:
        return {"rounds": 0, "aborted": aborted}
    wins = sum(r["result"] == "win" for r in records)
    losses = sum(r["result"] == "loss" for r in records)
    draws = n - wins - losses
    lo, hi = wilson(wins, n)
    survival = [r["survival"] for r in records if r["survival"] is not None]
    parts = {}
    for r in records:
        for k, v in r["parts"].items():
            parts[k] = parts.get(k, 0.0) + v
    out = {
        "rounds": n, "aborted": aborted, "win_rate": wins / n, "win_lo": lo, "win_hi": hi,
        "loss_rate": losses / n, "draw_rate": draws / n,
        "reward": float(np.mean([r["reward"] for r in records])),
        "survival": float(np.mean(survival)) if survival else 0.0,
        "death_rate": sum(r["died"] for r in records) / n,
        "kills": sum(r["kills"] for r in records) / n,
        "steps": float(np.mean([r["steps"] for r in records])),
        # share of decisions that turn / hold the brake (the zig-zag habit)
        "turn_rate": sum(r.get("turns", 0) for r in records) / max(1, sum(r["steps"] for r in records)),
        "brake_rate": sum(r.get("brakes", 0) for r in records) / max(1, sum(r["steps"] for r in records)),
    }
    for k, v in sorted(parts.items()):
        out[f"r_{k}"] = v / n
    return out


def by_opponent(records):
    groups = {}
    for r in records:
        groups.setdefault(r["opponent"], []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def format_summary(name, s):
    if not s.get("rounds"):
        return f"{name}: no rounds"
    return (f"{name}: {s['rounds']} rounds  win {s['win_rate']:.1%} [{s['win_lo']:.1%}, {s['win_hi']:.1%}]"
            f"  draw {s['draw_rate']:.1%}  loss {s['loss_rate']:.1%}  survival {s['survival']:.1f}s"
            f"  kills {s['kills']:.2f}  turns {s.get('turn_rate', 0):.0%}  reward {s['reward']:+.3f}")
