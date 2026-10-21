"""Exports a checkpoint as a small, self-contained model for serving or as a starting point.

    python -m fortrl.export runs/m4_6v6/checkpoints/latest.pt models/fort_6v6.pt --note "6v6, update 100"

Drops the optimizer and training state; keeps the network, its observation normalizer and
layout, plus a note on where it came from. Loads anywhere fortrl takes a --policy.
"""

import argparse
import datetime

import torch

from .policy import PolicyBundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)

    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bundle = PolicyBundle.load(args.checkpoint)
    torch.save({"policy": bundle.state_dict(),
                "meta": {"source": args.checkpoint, "note": args.note,
                         "state": source.get("state", {}),
                         "exported": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}},
               args.out)
    print(f"{args.out}: obs size {bundle.spec.size}, {sum(p.numel() for p in bundle.model.parameters())} weights")


if __name__ == "__main__":
    main()
