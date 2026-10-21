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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replays")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", action="append", default=[], help="label=eval source")
    args = parser.parse_args(argv)

    data = json.loads(Path(args.replays).read_text())
    data["stages"] = {}
    for item in args.stage:
        label, source = item.split("=", 1)
        data["stages"][label] = stage_result(source)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    Path(args.out).write_text(TEMPLATE.read_text().replace("__REPLAY_DATA__", payload))
    print(args.out)


if __name__ == "__main__":
    main()
