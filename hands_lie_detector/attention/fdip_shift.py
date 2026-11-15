"""
FDIP-like brightness features under capture-pipeline transforms (stdlib slice).
PROPOSED, co-authored. Constructed frames only; readings, no interpretation.

Delivered request:

    stdlib slice of FDIP H4: compute FDIP-like features (Laplacian, gradient
    magnitude, squared gradient) on constructed frames; apply resize /
    quantize / re-encode where stdlib allows; log feature shift per transform

"FDIP-like": the feature names come from the request. FDIP-NET itself was not
read (REFERENCES/attention_as_sensor.md entry 1, PRESS_RELEASE), so nothing
here reproduces its features or claims to.

Features, per frame, on luminance Y (Rec.709 weights on encoded 0-255 values,
the same Y as physics.py):
  laplacian_abs_mean   mean |4-neighbour Laplacian| over interior pixels
  laplacian_var        variance of the Laplacian over interior pixels
  grad_mag_mean        mean central-difference gradient magnitude (interior)
  grad_sq_mean         mean squared gradient magnitude (interior)

Transforms:
  resize_down2_up2     2x2 block mean, then nearest-neighbour back to size
  resize_nearest_half  every second pixel, then nearest-neighbour back to size
  quantize_6bit / quantize_4bit / quantize_2bit   keep the top N bits
  reencode_png         write a PNG (zlib, filter 0) and decode it with image.py
  reencode_ppm_maxval15  write a P6 with maxval 15 and decode it
  reencode_jpeg        NOT_AVAILABLE: the standard library has no JPEG codec

Shift per feature = after - before, and relative shift = shift / before
(None when before is 0: a relative change from zero is not a number).
"""

import json
import random
import struct
import zlib

from . import image as I

STATUS = "PROPOSED; constructed frames; readings only, no interpretation"
FEATURES = ("laplacian_abs_mean", "laplacian_var", "grad_mag_mean", "grad_sq_mean")


def luma(frame):
    return [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in frame.rgb]


def features(frame):
    w, h = frame.width, frame.height
    y = luma(frame)
    lap, gm, gs = [], [], []
    for j in range(1, h - 1):
        for i in range(1, w - 1):
            c = y[j * w + i]
            l, r = y[j * w + i - 1], y[j * w + i + 1]
            u, d = y[(j - 1) * w + i], y[(j + 1) * w + i]
            lap.append(l + r + u + d - 4 * c)
            gx, gy = (r - l) / 2.0, (d - u) / 2.0
            s = gx * gx + gy * gy
            gs.append(s)
            gm.append(s ** 0.5)
    n = len(lap) or 1
    m = sum(lap) / n
    return {"laplacian_abs_mean": sum(abs(v) for v in lap) / n,
            "laplacian_var": sum((v - m) ** 2 for v in lap) / n,
            "grad_mag_mean": sum(gm) / n,
            "grad_sq_mean": sum(gs) / n}


def _frame(w, h, rgb, mode="constructed"):
    return I.Frame(w, h, rgb, sha256=None, input_mode=mode)


# ----------------------------------------------------------------- frames

def constructed_frames(size=48, seed=0):
    """Five frames, each built for one property. Labelled CONSTRUCTED."""
    rng = random.Random(seed)
    w = h = size
    flat = [(120, 120, 120)] * (w * h)
    ramp = [(int(255 * x / (w - 1)),) * 3 for y in range(h) for x in range(w)]
    edge = [((230, 180, 160) if x >= w // 2 else (40, 30, 30)) for y in range(h) for x in range(w)]
    # the same edge one column over: an even column lines up with every 2x2 block
    # boundary, so a resize cannot touch it; an odd column does not
    edge_odd = [((230, 180, 160) if x >= w // 2 + 1 else (40, 30, 30))
                for y in range(h) for x in range(w)]
    tex = []
    for y in range(h):
        for x in range(w):
            base = 150 + (30 if (x // 3 + y // 3) % 2 else -30)
            v = max(0, min(255, base + rng.randint(-20, 20)))
            tex.append((v, int(v * 0.8), int(v * 0.7)))
    return {"flat": _frame(w, h, flat), "ramp": _frame(w, h, ramp),
            "step_edge": _frame(w, h, edge), "step_edge_odd": _frame(w, h, edge_odd),
            "fine_texture": _frame(w, h, tex)}


# ------------------------------------------------------------- transforms

def resize_down2_up2(f):
    w, h = f.width, f.height
    out = list(f.rgb)
    for j in range(0, h - 1, 2):
        for i in range(0, w - 1, 2):
            px = [f.pixel(i + a, j + b) for a in (0, 1) for b in (0, 1)]
            m = tuple(round(sum(p[k] for p in px) / 4) for k in range(3))
            for a in (0, 1):
                for b in (0, 1):
                    out[(j + b) * w + i + a] = m
    return _frame(w, h, out)


def resize_nearest_half(f):
    w, h = f.width, f.height
    out = [f.pixel((i // 2) * 2, (j // 2) * 2) for j in range(h) for i in range(w)]
    return _frame(w, h, out)


def quantize(bits):
    mask = (0xFF << (8 - bits)) & 0xFF

    def q(f):
        return _frame(f.width, f.height, [tuple(c & mask for c in p) for p in f.rgb])
    return q


def encode_png(f):
    raw = b"".join(b"\x00" + bytes(c for p in f.rgb[j * f.width:(j + 1) * f.width] for c in p)
                   for j in range(f.height))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", f.width, f.height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def reencode_png(f):
    w, h, rgb, _ = I.decode_png(encode_png(f))
    return _frame(w, h, rgb, "decoded:PNG")


def reencode_ppm_maxval15(f):
    hdr = f"P6\n{f.width} {f.height}\n15\n".encode()
    body = bytes(round(c * 15 / 255) for p in f.rgb for c in p)
    w, h, rgb, _ = I.decode_ppm(hdr + body)
    return _frame(w, h, rgb, "decoded:PPM")


TRANSFORMS = {
    "resize_down2_up2": resize_down2_up2,
    "resize_nearest_half": resize_nearest_half,
    "quantize_6bit": quantize(6),
    "quantize_4bit": quantize(4),
    "quantize_2bit": quantize(2),
    "reencode_png": reencode_png,
    "reencode_ppm_maxval15": reencode_ppm_maxval15,
    "reencode_jpeg": None,
}
NOT_AVAILABLE = {"reencode_jpeg": "the Python standard library has no JPEG codec"}


def shift(before, after):
    out = {}
    for k in FEATURES:
        d = after[k] - before[k]
        out[k] = {"before": round(before[k], 4), "after": round(after[k], 4),
                  "shift": round(d, 4),
                  "relative": None if before[k] == 0 else round(d / before[k], 4)}
    return out


def run(size=48, seed=0):
    frames = constructed_frames(size, seed)
    rows = []
    for fname, f in frames.items():
        base = features(f)
        for tname, t in TRANSFORMS.items():
            if t is None:
                rows.append({"frame": fname, "transform": tname, "state": "NOT_AVAILABLE",
                             "reason": NOT_AVAILABLE[tname]})
                continue
            rows.append({"frame": fname, "transform": tname, "state": "RUN",
                         "features": shift(base, features(t(f)))})
    return {"status": STATUS, "frames": "CONSTRUCTED (flat, ramp, step_edge, step_edge_odd, fine_texture)",
            "size": size, "seed": seed, "features": list(FEATURES), "rows": rows}


def render(rep):
    out = [f"FDIP-like feature shift ({rep['status']})",
           f"frames: {rep['frames']}  size {rep['size']}  seed {rep['seed']}",
           "relative = shift / before; '--' where before is 0", ""]
    hdr = f"{'frame':13} {'transform':22} " + " ".join(f"{k:>19}" for k in FEATURES)
    out.append(hdr)
    for r in rep["rows"]:
        if r["state"] != "RUN":
            out.append(f"{r['frame']:13} {r['transform']:22} {r['state']}: {r['reason']}")
            continue
        cells = []
        for k in FEATURES:
            v = r["features"][k]
            rel = "--" if v["relative"] is None else f"{v['relative']:+.3f}"
            cells.append(f"{v['shift']:+9.3f} ({rel:>7})")
        out.append(f"{r['frame']:13} {r['transform']:22} " + " ".join(cells))
    return "\n".join(out)


def main(argv=None):
    import argparse
    import sys
    p = argparse.ArgumentParser(description="FDIP-like feature shift on constructed frames")
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json")
    a = p.parse_args(argv)
    rep = run(a.size, a.seed)
    print(render(rep))
    if a.json:
        import os
        if os.path.exists(a.json):
            print(f"REFUSED: {a.json} exists", file=sys.stderr)
            return 2
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
