"""Plots training and evaluation curves of a run.

    python -m fortrl.plot runs/first            # writes runs/first/progress.png
"""

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(args.run_dir)
    metrics = read_jsonl(run / "metrics.jsonl")
    evals = read_jsonl(run / "eval.jsonl")
    if not metrics:
        raise SystemExit(f"no metrics in {run}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0][0]
    opponents = sorted({o for e in evals for o in e["results"]})
    for opp in opponents:
        pts = [(e["steps"], e["results"][opp]) for e in evals if opp in e["results"]]
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1]["win_rate"] for p in pts], marker="o", label=f"win vs {opp}")
        ax.fill_between(xs, [p[1]["win_lo"] for p in pts], [p[1]["win_hi"] for p in pts], alpha=0.2)
        ax.plot(xs, [p[1]["loss_rate"] for p in pts], linestyle="--", label=f"loss vs {opp}")
    ax.set_title("evaluation (separate matches, 95% CI)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    def train_series(key):
        out = {}
        for m in metrics:
            for opp, s in m["train"].items():
                if s.get("rounds"):
                    out.setdefault(opp, ([], []))
                    out[opp][0].append(m["steps"])
                    out[opp][1].append(s.get(key, 0.0))
        return out

    for ax, key, title in [(axes[0][1], "survival", "train: survival time (s)"),
                           (axes[0][2], "reward", "train: reward per round"),
                           (axes[1][0], "win_rate", "train: win rate")]:
        for opp, (xs, ys) in train_series(key).items():
            ax.plot(xs, ys, label=opp)
        ax.set_title(title)
        ax.legend(fontsize=8)

    steps = [m["steps"] for m in metrics]
    ax = axes[1][1]
    ax.plot(steps, [m["entropy"] for m in metrics], label="entropy")
    ax.plot(steps, [m["explained_variance"] for m in metrics], label="value expl. var.")
    ax.set_title("optimization")
    ax.legend(fontsize=8)

    ax = axes[1][2]
    parts = sorted({k for m in metrics for s in m["train"].values() for k in s if k.startswith("r_")})
    def weighted(m, part):
        rounds = sum(s.get("rounds", 0) for s in m["train"].values())
        return sum(s.get(part, 0.0) * s.get("rounds", 0) for s in m["train"].values()) / max(rounds, 1)

    for part in parts:
        ax.plot(steps, [weighted(m, part) for m in metrics], label=part[2:])
    ax.set_title("train: reward components per round")
    ax.legend(fontsize=8)

    for row in axes:
        for ax in row:
            ax.set_xlabel("learner steps")
            ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.out) if args.out else run / "progress.png"
    fig.savefig(out, dpi=100)
    print(out)


if __name__ == "__main__":
    main()
