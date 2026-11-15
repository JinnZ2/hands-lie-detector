"""
Attention against physics: landings, controls, the four cells, divergence.

Inputs are one frame and separate observer logs. Outputs are numbers and
labels; nothing here interprets them.

Grid. The frame is tiled into G x G cells                              [CHOICE 5]
(default 8; the last row and column absorb remainders). A cell is
ATTENDED by an observer when any of that observer's mapped regions
overlaps it.

Physics signal. A cell or region has a SIGNAL when its edge_strength is at
least SIGNAL_K times the frame's edge fraction                         [CHOICE 6]
(default 1.5). If the frame's edge fraction is 0, any edge counts.

Four labels per cell, per observer, never pooled across observers:
    attended + signal      CALIBRATED_POINTER
    attended, no signal    BIAS
    signal, not attended   BLIND_SPOT
    neither                NEITHER
In coordinates-only mode every cell is NOT_EVALUABLE.

Controls, per observer, seeded:
    random regions   the observer's mapped regions, same sizes, placed at
                     random; statistic = mean edge_strength over them.
                     Reported: observed value, null mean, share of draws >=
                     observed.
    shuffled order   Spearman rank correlation between listing order and
                     edge_strength; null = the same with the order shuffled.
                     Reported: rho, share of draws with |rho| >= observed.

Divergence (the atmosphere map): per cell, which observers attended it. Cells
attended by some observers and not others are listed with both sets.
"""

import random

from . import physics as P

SIGNAL_K = 1.5
GRID = 8
DRAWS = 200

LABELS = ("CALIBRATED_POINTER", "BIAS", "BLIND_SPOT", "NEITHER")


def cells(width, height, g):
    out = []
    for j in range(g):
        for i in range(g):
            x0, x1 = i * width // g, (i + 1) * width // g if i < g - 1 else width
            y0, y1 = j * height // g, (j + 1) * height // g if j < g - 1 else height
            if x1 > x0 and y1 > y0:
                out.append((f"r{j}c{i}", [x0, y0, x1, y1]))
    return out


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _clamp(r, w, h):
    return [max(0, min(r[0], w - 1)), max(0, min(r[1], h - 1)),
            max(1, min(r[2], w)), max(1, min(r[3], h))]


def has_signal(reading, frame_fraction, k=SIGNAL_K):
    e = reading.get("edge_strength")
    if not isinstance(e, (int, float)) or frame_fraction is None:
        return None
    return e > 0 if frame_fraction == 0 else e >= k * frame_fraction


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(a, b):
    """Rank correlation with average ranks for ties; None below 3 points or on a constant."""
    if len(a) < 3 or len(a) != len(b):
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return None if da == 0 or db == 0 else num / (da * db)


def read_observer(field, log, backlit=False, grid=GRID, k=SIGNAL_K, seed=0, draws=DRAWS,
                  with_controls=True):
    fr = field.frame
    ff = field.frame_edge_fraction() if fr.decoded else None
    mapped = [e for e in log["entries"] if e.get("region") is not None]
    landings = []
    for e in mapped:
        reg = _clamp(e["region"], fr.width, fr.height)
        rd = P.region_reading(field, reg, backlit)
        sig = has_signal(rd, ff, k)
        landings.append({"rank": e["rank"], "noticed": e["noticed"], "region": reg,
                         "physics": rd,
                         "label": (None if sig is None else
                                   "CALIBRATED_POINTER" if sig else "BIAS")})
    table = {}
    for cid, box in cells(fr.width, fr.height, grid):
        attended = any(_overlap(box, l["region"]) for l in landings)
        rd = P.region_reading(field, box, backlit)
        sig = has_signal(rd, ff, k)
        if sig is None:
            label = "NOT_EVALUABLE"
        elif attended:
            label = "CALIBRATED_POINTER" if sig else "BIAS"
        else:
            label = "BLIND_SPOT" if sig else "NEITHER"
        table[cid] = {"box": box, "attended": attended, "signal": sig, "label": label,
                      "edge_strength": rd.get("edge_strength")}
    counts = {lab: sum(1 for c in table.values() if c["label"] == lab)
              for lab in LABELS + ("NOT_EVALUABLE",)}
    return {"observer": log["observer"], "entries": len(log["entries"]),
            "unmapped_entries": len(log["entries"]) - len(mapped),
            "landings": landings, "cells": table, "cell_counts": counts,
            "controls": (controls(field, landings, seed, draws) if fr.decoded and with_controls else
                         {"state": "NOT_RUN", "reason": "controls not requested (bound sweep)"}
                         if fr.decoded else
                         {"state": "NOT_EVALUABLE", "reason": "coordinates-only: no pixels"})}


def controls(field, landings, seed=0, draws=DRAWS):
    fr = field.frame
    es = [l["physics"]["edge_strength"] for l in landings]
    if not landings or any(not isinstance(v, (int, float)) for v in es):
        return {"state": "NOT_EVALUABLE", "reason": "no mapped region with a reading"}
    rng = random.Random(seed)
    observed = sum(es) / len(es)
    null = []
    for _ in range(draws):
        vals = []
        for l in landings:
            x0, y0, x1, y1 = l["region"]
            rw, rh = x1 - x0, y1 - y0
            nx = rng.randrange(0, fr.width - rw + 1)
            ny = rng.randrange(0, fr.height - rh + 1)
            v = P.region_reading(field, [nx, ny, nx + rw, ny + rh])["edge_strength"]
            vals.append(v if isinstance(v, (int, float)) else 0.0)
        null.append(sum(vals) / len(vals))
    order = [l["rank"] for l in landings]
    rho = spearman(order, es)
    perm = []
    if rho is not None:
        for _ in range(draws):
            sh = order[:]
            rng.shuffle(sh)
            r = spearman(sh, es)
            perm.append(abs(r) if r is not None else 0.0)
    return {
        "seed": seed, "draws": draws,
        "random_regions": {"observed_mean_edge_strength": observed,
                           "null_mean": sum(null) / len(null),
                           "share_of_draws_ge_observed": sum(v >= observed for v in null) / len(null)},
        "shuffled_order": {"spearman_order_vs_edge_strength": rho,
                           "share_of_draws_abs_ge_observed":
                               (sum(v >= abs(rho) for v in perm) / len(perm)) if perm else None},
    }


def divergence(readings):
    """Per cell, which observers attended it. Built from separate readings; no log is merged."""
    names = [f"{r['observer']['kind']}:{r['observer']['id']}" for r in readings]
    if not readings:
        return {"observers": [], "cells": {}, "split_cells": []}
    out, split = {}, []
    for cid in readings[0]["cells"]:
        who = [n for n, r in zip(names, readings) if r["cells"][cid]["attended"]]
        out[cid] = who
        if 0 < len(who) < len(readings):
            split.append({"cell": cid, "attended_by": who,
                          "not_attended_by": [n for n in names if n not in who]})
    return {"observers": names, "cells": out, "split_cells": split,
            "all_attended": sum(1 for w in out.values() if len(w) == len(readings)),
            "none_attended": sum(1 for w in out.values() if not w)}
