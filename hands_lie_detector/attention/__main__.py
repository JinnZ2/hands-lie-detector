"""python3 -m hands_lie_detector.attention --image f.png --log a.json [--log b.json] ..."""

import argparse
import json
import os
import sys

from . import image as I
from . import log as L
from .run import render, run


def main(argv=None):
    p = argparse.ArgumentParser(description="attention-as-sensor (PROPOSED): readings, no interpretation")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="PNG or PPM; JPEG/HEIC are refused (use --size)")
    src.add_argument("--size", nargs=2, type=int, metavar=("W", "H"),
                     help="coordinates-only mode: no pixels; physics reads UNMEASURED")
    p.add_argument("--image-sha256", help="coordinates-only: the frame's sha256, to check logs against")
    p.add_argument("--log", action="append", default=[], help="one observer's log (repeatable)")
    p.add_argument("--backlit", action="store_true",
                   help="frame is backlit: transmission (path opacity) is read; stays on device")
    p.add_argument("--small", action="store_true", help="downsample to <= 128 px on the long side")
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--signal-k", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="write the full result here (refuses an existing path)")
    a = p.parse_args(argv)
    try:
        frame = (I.load(a.image) if a.image else
                 I.coordinates_only(a.size[0], a.size[1], a.image_sha256))
        rep = run(frame, [L.read(x) for x in a.log], a.backlit, a.small, a.grid, a.signal_k, a.seed)
    except (I.UnsupportedImage, L.LogError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if a.json:
        if os.path.exists(a.json):
            print(f"REFUSED: {a.json} exists; results are not overwritten", file=sys.stderr)
            return 2
        with open(a.json, "x", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    print(render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
