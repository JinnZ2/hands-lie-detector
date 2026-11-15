"""
Attention logs: what an observer noticed, in order, before any task prompt.

One log per observer per image. Observers are Claude, other models, and the
human carrier. Logs are kept separate and are never merged: there is no
function that concatenates entries across observers, and `ObserverLogs`
refuses a second log for an observer it already holds.

A log is valid only if it was taken before any task prompt
(`before_task_prompt: true`). A list produced after the task was stated is a
reading of the task, not of the image, and is refused.

Each entry is {rank, noticed, region}: region is [x0, y0, x1, y1] in pixel
coordinates of the image as supplied (x1, y1 exclusive), or null when the
observer named something that was not mapped to a place. Unmapped entries are
counted and never placed.
"""

import json

OBSERVER_KINDS = ("claude", "model", "human")
SLOTS = ("claude", "other_models", "human")


class LogError(ValueError):
    pass


def validate(log, width, height, image_sha256=None):
    obs = log.get("observer") or {}
    if obs.get("kind") not in OBSERVER_KINDS:
        raise LogError(f"observer kind must be one of {OBSERVER_KINDS}, got {obs.get('kind')!r}")
    if not obs.get("id"):
        raise LogError("observer id missing (name the model or the person's role, not a merge)")
    if log.get("before_task_prompt") is not True:
        raise LogError("before_task_prompt must be true: a list taken after the task prompt "
                       "reads the task, not the image")
    if image_sha256 and log.get("image_sha256") and log["image_sha256"] != image_sha256:
        raise LogError("image_sha256 does not match the image: this log is for another frame")
    entries = log.get("entries")
    if not isinstance(entries, list):
        raise LogError("entries must be a list (an empty list is legal: nothing noticed)")
    ranks = [e.get("rank") for e in entries]
    if ranks != list(range(1, len(entries) + 1)):
        raise LogError(f"ranks must be 1..n in listing order, got {ranks}")
    for e in entries:
        if not e.get("noticed"):
            raise LogError(f"entry {e.get('rank')}: 'noticed' is empty")
        r = e.get("region")
        if r is None:
            continue
        if (not isinstance(r, list) or len(r) != 4
                or not all(isinstance(v, int) for v in r)):
            raise LogError(f"entry {e.get('rank')}: region must be [x0, y0, x1, y1] integers or null")
        x0, y0, x1, y1 = r
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise LogError(f"entry {e.get('rank')}: region {r} outside the {width}x{height} image "
                           f"or empty")
    return log


def slot_of(log):
    k = log["observer"]["kind"]
    return {"claude": "claude", "model": "other_models", "human": "human"}[k]


class ObserverLogs:
    """Holds separate logs. Never merges them."""

    def __init__(self):
        self._logs = {}

    def add(self, log):
        key = (log["observer"]["kind"], log["observer"]["id"])
        if key in self._logs:
            raise LogError(f"a log for observer {key} is already held; logs are not merged "
                           f"or replaced")
        self._logs[key] = log

    def items(self):
        return list(self._logs.items())

    def slots(self):
        """Every slot is present; a slot with no log is null, never filled in."""
        out = {s: None for s in SLOTS}
        for (kind, oid), log in self._logs.items():
            s = slot_of(log)
            if s == "other_models":
                out[s] = (out[s] or []) + [oid]
            else:
                out[s] = oid
        return out


def read(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def scale(log, factor):
    """--small: map regions onto the downsampled frame (floor/ceil keep the box covering)."""
    if factor == 1:
        return log
    out = json.loads(json.dumps(log))
    for e in out["entries"]:
        r = e.get("region")
        if r is not None:
            x0, y0, x1, y1 = r
            e["region"] = [x0 // factor, y0 // factor, -(-x1 // factor), -(-y1 // factor)]
    return out
