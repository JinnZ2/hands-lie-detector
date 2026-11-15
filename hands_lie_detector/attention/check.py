"""
Check a stored attention log of either kind (free or prompted).

    python3 -m hands_lie_detector.attention.check FILE [--size W H]

A prompted log passes here and is still never a probe: run and sweep refuse it.
"""

from .log import LogError, log_kind, read, validate_stored


def main(argv=None):
    """Check a stored log of either kind."""
    import argparse
    import sys
    p = argparse.ArgumentParser(description="check a stored attention log (free or prompted)")
    p.add_argument("file")
    p.add_argument("--size", nargs=2, type=int, metavar=("W", "H"))
    a = p.parse_args(argv)
    w, h = a.size if a.size else (None, None)
    try:
        lg = validate_stored(read(a.file), w, h)
    except LogError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    k = log_kind(lg)
    o = lg["observer"]
    print(f"VALID  {o['kind']}:{o['id']}  log_kind {k}  entries {len(lg['entries'])}  "
          f"unmapped {sum(e.get('region') is None for e in lg['entries'])}  "
          f"per_photo_order {lg.get('per_photo_order')}  image_sha256 {lg.get('image_sha256')}")
    if k == "prompted":
        print("prompted: stored for comparison; never used as a probe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
