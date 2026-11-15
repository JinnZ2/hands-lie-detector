"""One run: a frame, separate logs, the readings. No interpretation is produced."""

import json

from . import image as I
from . import log as L
from . import maps as M
from . import physics as P

PRIVACY = ("aggregates only: no pixels, crops or per-pixel maps are written. Backlit frames "
           "show the vein pattern (backlit-transmission.md, rule zero); they stay on device "
           "and only region aggregates leave this function.")


def run(frame, logs, backlit=False, small=False, grid=M.GRID, k=M.SIGNAL_K, seed=0,
        draws=M.DRAWS):
    factor = 1
    if small:
        frame, factor = I.small(frame)
    held = L.ObserverLogs()
    for lg in logs:
        L.validate(lg, frame.width * factor, frame.height * factor, frame.sha256)
        held.add(lg)
    field = P.Field(frame)
    readings = [M.read_observer(field, L.scale(lg, factor), backlit, grid, k, seed, draws)
                for _, lg in held.items()]
    return {
        "status": "PROPOSED; readings only, no interpretation",
        "input_mode": frame.input_mode,
        "image_sha256": frame.sha256,
        "frame": {"width": frame.width, "height": frame.height, "backlit": backlit},
        "bounds": {**frame.bounds, **P.bounds(), "grid": grid, "SIGNAL_K": k,
                   "draws": draws, "seed": seed, "small_factor": factor,
                   "frame_edge_fraction": field.frame_edge_fraction() if frame.decoded else
                   P.UNMEASURED},
        "observer_slots": held.slots(),
        "observers": readings,
        "divergence": M.divergence(readings),
        "privacy": PRIVACY,
    }


def render(rep):
    out = [f"attention-as-sensor ({rep['status']})",
           f"input: {rep['input_mode']}  {rep['frame']['width']}x{rep['frame']['height']}"
           f"  backlit={rep['frame']['backlit']}  sha256 {str(rep['image_sha256'])[:12]}",
           "bounds: " + json.dumps(rep["bounds"], sort_keys=True),
           "observer slots: " + json.dumps(rep["observer_slots"]),
           f"privacy: {rep['privacy']}", ""]
    for r in rep["observers"]:
        o = r["observer"]
        out.append(f"== observer {o['kind']}:{o['id']}  entries {r['entries']}"
                   f"  unmapped {r['unmapped_entries']}")
        for l in r["landings"]:
            ph = l["physics"]
            out.append(f"  #{l['rank']} {l['region']} {l['label']}  "
                       f"grad={ph.get('luminance_gradient')} edge={ph.get('edge_strength')} "
                       f"spec={ph.get('specular')} diff={ph.get('diffuse')} "
                       f"r/b={ph.get('transmission_r_over_b')}  ({l['noticed']})")
        out.append("  cells: " + json.dumps(r["cell_counts"]))
        out.append("  controls: " + json.dumps(r["controls"], sort_keys=True))
    d = rep["divergence"]
    out.append("")
    out.append(f"divergence: observers {d['observers']}  all attended {d.get('all_attended')}"
               f"  none attended {d.get('none_attended')}  split cells {len(d['split_cells'])}")
    for s in d["split_cells"]:
        out.append(f"  {s['cell']}: attended by {s['attended_by']}; not by {s['not_attended_by']}")
    return "\n".join(out)
