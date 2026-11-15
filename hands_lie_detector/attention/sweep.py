"""
Bound sweep (Card 08 rule in study-aperture): test every bound the labels rest
on, not only the suspected one. Readings only, no interpretation.

Swept bounds, all combinations (27 by default):
    EDGE_T     light-mass boundary threshold     16, 32, 64
    SIGNAL_K   signal multiple of the frame's edge fraction   1.0, 1.5, 2.0
    GRID       cells per side                    6, 8, 12

Reported per observer:
    landings   each mapped entry's label at every setting. HOLDS when one label
               across all settings; SHIFTS otherwise, with every setting at
               which it differs from the default (32, 1.5, 8).
               Landings do not depend on GRID.
    cells      per GRID, label counts at each (EDGE_T, SIGNAL_K), and the share
               of cells whose label holds across those 9 settings. Cells are
               not compared across grids: the cell geometry changes.
    lattice    a grid-independent view. Points on a regular lattice (at most
               LATTICE_MAX per side, stated) take the label of the cell they
               fall in. Per point: the set of labels it gets across all settings.
               Reported: count of points holding each label at every setting,
               and the count that shift.
Controls are not run in the sweep (they do not set labels).
"""

import itertools
import json

from . import maps as M
from .run import run

EDGE_TS = (16.0, 32.0, 64.0)
KS = (1.0, 1.5, 2.0)
GRIDS = (6, 8, 12)
DEFAULT = (32.0, 1.5, 8)
LATTICE_MAX = 64


def _cell_of(x, y, width, height, g):
    i = min(g - 1, x * g // width)
    j = min(g - 1, y * g // height)
    return f"r{j}c{i}"


def sweep(frame, logs, backlit=False, small=False, edge_ts=EDGE_TS, ks=KS, grids=GRIDS):
    settings = list(itertools.product(edge_ts, ks, grids))
    reps = {s: run(frame, logs, backlit, small, grid=s[2], k=s[1], edge_t=s[0],
                   with_controls=False) for s in settings}
    first = reps[settings[0]]
    w, h = first["frame"]["width"], first["frame"]["height"]
    step_x, step_y = max(1, -(-w // LATTICE_MAX)), max(1, -(-h // LATTICE_MAX))
    lattice = [(x, y) for y in range(step_y // 2, h, step_y) for x in range(step_x // 2, w, step_x)]
    default = DEFAULT if DEFAULT in reps else settings[0]
    out = {"status": "PROPOSED; bound sweep, readings only, no interpretation",
           "input_mode": first["input_mode"], "image_sha256": first["image_sha256"],
           "frame": first["frame"],
           "bounds": {"EDGE_T": list(edge_ts), "SIGNAL_K": list(ks), "GRID": list(grids),
                      "settings": len(settings), "default": list(default),
                      "lattice_points": len(lattice), "lattice_step": [step_x, step_y],
                      "small_factor": first["bounds"]["small_factor"]},
           "observer_slots": first["observer_slots"], "observers": []}
    if not first["input_mode"].startswith("decoded:"):
        out["state"] = "NOT_EVALUABLE"
        out["reason"] = "coordinates-only: no pixels, so no label to sweep"
        return out
    for oi, obs in enumerate(first["observers"]):
        name = f"{obs['observer']['kind']}:{obs['observer']['id']}"
        landings = []
        for li, l in enumerate(obs["landings"]):
            per = {s: reps[s]["observers"][oi]["landings"][li]["label"] for s in settings}
            labels = sorted(set(per.values()), key=str)
            row = {"rank": l["rank"], "noticed": l["noticed"], "labels": labels,
                   "state": "HOLDS" if len(labels) == 1 else "SHIFTS"}
            if len(labels) > 1:
                row["differs_from_default_at"] = [
                    {"EDGE_T": s[0], "SIGNAL_K": s[1], "label": per[s]}
                    for s in settings if s[2] == default[2] and per[s] != per[default]]
            landings.append(row)
        cells = {}
        for g in grids:
            sub = [s for s in settings if s[2] == g]
            tables = {s: reps[s]["observers"][oi]["cells"] for s in sub}
            ids = list(tables[sub[0]])
            held = {lab: 0 for lab in M.LABELS}
            shifting = 0
            for cid in ids:
                labs = {tables[s][cid]["label"] for s in sub}
                if len(labs) == 1:
                    held[labs.pop()] += 1
                else:
                    shifting += 1
            cells[str(g)] = {
                "cells": len(ids),
                "counts": [{"EDGE_T": s[0], "SIGNAL_K": s[1],
                            **reps[s]["observers"][oi]["cell_counts"]} for s in sub],
                "held_across_edge_and_k": held, "shifting": shifting}
        held = {lab: 0 for lab in M.LABELS}
        shifting = 0
        for x, y in lattice:
            labs = {reps[s]["observers"][oi]["cells"][_cell_of(x, y, w, h, s[2])]["label"]
                    for s in settings}
            if len(labs) == 1:
                held[labs.pop()] += 1
            else:
                shifting += 1
        out["observers"].append({
            "observer": name, "landings": landings, "cells_by_grid": cells,
            "lattice": {"held_across_all_settings": held, "shifting": shifting,
                        "points": len(lattice)},
            "landing_labels_holding": sorted({r["labels"][0] for r in landings
                                              if r["state"] == "HOLDS"}, key=str)})
    return out


def render(rep):
    out = [f"bound sweep ({rep['status']})",
           f"input: {rep['input_mode']}  {rep['frame']['width']}x{rep['frame']['height']}",
           "bounds: " + json.dumps(rep["bounds"], sort_keys=True)]
    if rep.get("state"):
        return "\n".join(out + [f"{rep['state']}: {rep['reason']}"])
    for o in rep["observers"]:
        out.append(f"== {o['observer']}")
        for l in o["landings"]:
            out.append(f"  landing #{l['rank']}: {l['state']} {l['labels']}")
            for d in l.get("differs_from_default_at", []):
                out.append(f"      at EDGE_T {d['EDGE_T']} SIGNAL_K {d['SIGNAL_K']}: {d['label']}")
        for g, c in o["cells_by_grid"].items():
            out.append(f"  grid {g} ({c['cells']} cells): held across EDGE_T x SIGNAL_K "
                       f"{json.dumps(c['held_across_edge_and_k'])}, shifting {c['shifting']}")
            for row in c["counts"]:
                out.append("      " + json.dumps(row))
        lat = o["lattice"]
        out.append(f"  lattice ({lat['points']} points, all settings): held "
                   f"{json.dumps(lat['held_across_all_settings'])}, shifting {lat['shifting']}")
    return "\n".join(out)


def main(argv=None):
    import argparse, os, sys
    from . import image as I
    from . import log as L
    p = argparse.ArgumentParser(description="attention-as-sensor bound sweep (readings only)")
    p.add_argument("--image", required=True)
    p.add_argument("--log", action="append", default=[])
    p.add_argument("--backlit", action="store_true")
    p.add_argument("--small", action="store_true")
    p.add_argument("--json")
    a = p.parse_args(argv)
    try:
        rep = sweep(I.load(a.image), [L.read(x) for x in a.log], a.backlit, a.small)
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
    import sys
    sys.exit(main())
