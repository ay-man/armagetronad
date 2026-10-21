"""Builds a self-contained HTML replay viewer from fortrl.record output.

    python -m fortrl.replay_page replays.json --out replays.html \\
        --stage "Random play=runs/baseline_random.json" \\
        --stage "Run A, update 50=runs/m1_builtin:50"

--stage attaches evaluation results vs the built-in AI to a label: either an evaluate
--out JSON file, or <run dir>:<update> to pick a line of that run's eval.jsonl.
"""

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("replay_viewer.html")


def stage_result(spec):
    if ":" in spec and not spec.endswith(".json"):
        run, update = spec.rsplit(":", 1)
        for line in (Path(run) / "eval.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["update"] == int(update):
                return {**row["results"]["builtin"], "trained_steps": row["steps"]}
        raise SystemExit(f"no evaluation at update {update} in {run}/eval.jsonl")
    return json.loads(Path(spec).read_text())["results"]["builtin"]


def final_steps(run):
    lines = (Path(run) / "metrics.jsonl").read_text().splitlines()
    return json.loads(lines[-1])["steps"] if lines else 0


def run_progress(spec, compare):
    """'label=run_dir[+ancestor...]': evaluation history, x offset by the ancestors' training."""
    label, rest = spec.split("=", 1)
    run, *parents = rest.split("+")
    offset = sum(final_steps(p) for p in parents)
    points = []
    for line in (Path(run) / "eval.jsonl").read_text().splitlines():
        row = json.loads(line)
        point = {"x": offset + row["steps"]}
        for key, name in (("builtin", "builtin"), (compare, "compare")):
            r = row["results"].get(key)
            if r and r.get("rounds"):
                point[name] = {"w": r["win_rate"], "lo": r["win_lo"], "hi": r["win_hi"], "n": r["rounds"],
                               "turn": r.get("turn_rate")}
        points.append(point)
    return {"label": label, "start": offset, "points": points}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replays")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", action="append", default=[], help="label=eval source")
    parser.add_argument("--run", action="append", default=[],
                        help="label=run_dir[+parent+grandparent...]: plot this run's evaluations after its ancestors' training")
    parser.add_argument("--compare", default="", help="eval opponent key for the head-to-head chart, e.g. ckpt:<path>")
    parser.add_argument("--compare-label", default="")
    parser.add_argument("--turn-ref", default="", help="label=evaluate --out JSON: reference turn rate vs built-in AI")
    parser.add_argument("--title", default="Fortress Bot Replays")
    parser.add_argument("--lede", default="Recorded Fortress matches between the learning bots and the game's built-in AI. "
                        "The bots started with no strategy at all; everything they do, they learned from playing.")
    args = parser.parse_args(argv)

    data = json.loads(Path(args.replays).read_text())
    data["stages"] = {}
    for item in args.stage:
        label, source = item.split("=", 1)
        data["stages"][label] = stage_result(source)
    if args.run:
        data["progress"] = {"runs": [run_progress(r, args.compare) for r in args.run],
                            "compare_label": args.compare_label}
        if args.turn_ref:
            label, path = args.turn_ref.split("=", 1)
            ref = json.loads(Path(path).read_text())["results"]["builtin"]
            data["progress"]["turn_ref"] = {"label": label, "value": ref["turn_rate"]}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    import html
    page = (TEMPLATE.read_text().replace("__TITLE__", html.escape(args.title))
            .replace("__LEDE__", html.escape(args.lede)).replace("__REPLAY_DATA__", payload))
    Path(args.out).write_text(page)
    print(args.out)


if __name__ == "__main__":
    main()
