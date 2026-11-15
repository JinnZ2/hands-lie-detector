"""
Build an attention log from an order of noticed items whose regions are not
drawn yet. Readings only; nothing is placed.

Every entry gets region null (unmapped: counted, never placed). The log carries
its provenance. A CARRIED log came from a transcript, not from a run, and
before_task_prompt is left null: it is set to true by hand only when the
transcript shows the list came before any task prompt. The validator refuses
the log until then.

    python3 -m hands_lie_detector.attention.template --id <model> \\
        --source "chat transcript, 2026-11-15 morning" --item "first thing" --item "second" ...
"""

import argparse
import json
import sys


def build(kind, observer_id, items, source, grade="CARRIED", image_sha256=None):
    return {
        "observer": {"kind": kind, "id": observer_id},
        "image_sha256": image_sha256,
        "before_task_prompt": None,
        "provenance": {"source": source, "grade": grade,
                       "regions": "void until drawn",
                       "before_task_prompt": "unset: set true only if the source shows the "
                                             "list came before any task prompt"},
        "entries": [{"rank": i + 1, "noticed": t, "region": None} for i, t in enumerate(items)],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="attention log from an order, regions void")
    p.add_argument("--kind", default="claude", choices=("claude", "model", "human"))
    p.add_argument("--id", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--grade", default="CARRIED")
    p.add_argument("--item", action="append", default=[], help="one noticed item, in order")
    p.add_argument("--image-sha256")
    a = p.parse_args(argv)
    if not a.item:
        print("REFUSED: no --item given; an order with no items is not a log", file=sys.stderr)
        return 2
    json.dump(build(a.kind, a.id, a.item, a.source, a.grade, a.image_sha256), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
