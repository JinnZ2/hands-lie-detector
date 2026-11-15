"""
Frame record: which uploaded file is which frame, which log kinds apply to it,
and the sha256 of the file it was decoded from.

attention_runs/frames.json holds one entry per frame label. Images are never
committed; only labels, flags and hashes are.

Rules:
- A log's `frame_label` must name a frame in the record, and its `log_kind`
  must be one the frame allows (a frame marked prompted-only takes no free log).
- `image_sha256` is filled only from a file the decoder actually decoded, and
  `image_sha256_source` logs which format that was (PNG, PPM, or JPEG once the
  decoder handles JPEG). A file the decoder refuses gives no hash here.
- `fill` writes the hash into the frame record and into every log in the same
  folder carrying that frame_label; it refuses to overwrite a different hash.
- `backlit: true` frames: the physics run stays on device (backlit-transmission.md,
  rule zero). Only the JSON aggregates leave it.

    python3 -m hands_lie_detector.attention.frames check  [--dir attention_runs]
    python3 -m hands_lie_detector.attention.frames fill --frame photo1 --image PATH [--dir attention_runs]
"""

import argparse
import json
import pathlib
import sys

from . import image as I
from . import log as L

FRAMES_FILE = "frames.json"


def load(dirpath):
    return json.loads((pathlib.Path(dirpath) / FRAMES_FILE).read_text())


def check_log(log, frames):
    label = log.get("frame_label")
    if label is None:
        raise L.LogError("log has no frame_label; the frame record cannot place it")
    f = frames["frames"].get(label)
    if f is None:
        raise L.LogError(f"frame_label {label!r} is not in the frame record")
    k = L.log_kind(log)
    if k not in f["allowed_log_kinds"]:
        raise L.LogError(f"frame {label} ({f.get('source_file')}) allows {f['allowed_log_kinds']}, "
                         f"not a {k} log")
    if log.get("image_sha256") and f.get("image_sha256") and log["image_sha256"] != f["image_sha256"]:
        raise L.LogError(f"log image_sha256 differs from the frame record for {label}")
    return f


def logs_in(dirpath):
    out = []
    for p in sorted(pathlib.Path(dirpath).glob("*.json")):
        if p.name == FRAMES_FILE:
            continue
        d = json.loads(p.read_text())
        if isinstance(d, dict) and "observer" in d and "entries" in d:
            out.append((p, d))
    return out


def check_dir(dirpath):
    frames = load(dirpath)
    rows = []
    for p, lg in logs_in(dirpath):
        L.validate_stored(lg)
        f = check_log(lg, frames)
        rows.append((p.name, lg["frame_label"], f.get("source_file"), L.log_kind(lg)))
    return frames, rows


def fill(dirpath, label, image_path):
    dirpath = pathlib.Path(dirpath)
    frames = load(dirpath)
    f = frames["frames"].get(label)
    if f is None:
        raise L.LogError(f"frame {label!r} is not in the frame record")
    try:
        frame = I.load(image_path)
    except I.UnsupportedImage as e:
        raise L.LogError(f"not filled: the decoder refused the file ({e}); the hash is taken "
                         f"only from a file that decodes")
    fmt = frame.input_mode.split(":", 1)[1]
    for holder in [f] + [lg for _, lg in logs_in(dirpath) if lg.get("frame_label") == label]:
        old = holder.get("image_sha256")
        if old is not None and old != frame.sha256:
            raise L.LogError(f"{label}: a different image_sha256 is already recorded; not overwritten")
    f["image_sha256"], f["image_sha256_source"] = frame.sha256, fmt
    (dirpath / FRAMES_FILE).write_text(json.dumps(frames, indent=1) + "\n")
    n = 0
    for p, lg in logs_in(dirpath):
        if lg.get("frame_label") == label:
            lg["image_sha256"], lg["image_sha256_source"] = frame.sha256, fmt
            L.validate_stored(lg)
            p.write_text(json.dumps(lg, indent=1) + "\n")
            n += 1
    return frame.sha256, fmt, n


def main(argv=None):
    p = argparse.ArgumentParser(description="attention frame record")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--dir", default="attention_runs")
    fl = sub.add_parser("fill")
    fl.add_argument("--dir", default="attention_runs")
    fl.add_argument("--frame", required=True)
    fl.add_argument("--image", required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "check":
            frames, rows = check_dir(a.dir)
            for lbl, f in frames["frames"].items():
                print(f"frame {lbl}: {f.get('source_file')}  set {f.get('set')}  "
                      f"allows {f['allowed_log_kinds']}  backlit {f.get('backlit')}  "
                      f"sha256 {f.get('image_sha256')} ({f.get('image_sha256_source')})")
            for name, lbl, src, k in rows:
                print(f"  log {name}: frame {lbl} ({src}), {k}")
            return 0
        sha, fmt, n = fill(a.dir, a.frame, a.image)
        print(f"filled {a.frame}: sha256 {sha} from decoded {fmt}; {n} log(s) updated")
        return 0
    except L.LogError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
