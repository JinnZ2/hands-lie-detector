"""
Physics layer: what sits under a region, from pixel values only.

Every number here is a property of the encoded pixels, not of tissue. Each
reading is returned with the bounds it was taken under, and none is
interpreted.

    luminance        Y = 0.2126 R + 0.7152 G + 0.0722 B on encoded 0-255
                     values (no gamma linearisation)                     [CHOICE 1]
    gradient         central-difference magnitude of Y, interior pixels   [CHOICE 2]
    luminance_gradient   mean gradient over the region
    edge_strength    share of region pixels with gradient >= EDGE_T: the
                     light-mass boundaries                               [CHOICE 3]
    specular         share of pixels with Y >= SPEC_Y and max-min channel
                     <= SPEC_C (bright and unsaturated)                  [CHOICE 4]
    diffuse          share of pixels with DARK_Y < Y < SPEC_Y
    transmission     mean R / mean B. Read only on a frame declared backlit;
                     it is path opacity (backlit-transmission.md: everything on
                     the light path, not the callus plate). NOT_APPLICABLE on
                     other frames; None when mean B is 0.

A region in coordinates-only mode has every reading UNMEASURED.
"""

EDGE_T = 32.0
SPEC_Y = 230.0
SPEC_C = 30.0
DARK_Y = 20.0

UNMEASURED = "UNMEASURED"
NOT_APPLICABLE = "NOT_APPLICABLE"


def bounds(edge_t=EDGE_T):
    return {"luminance": "Y = 0.2126R + 0.7152G + 0.0722B on encoded values (no linearisation)",
            "gradient": "central differences on interior pixels; border pixels carry none",
            "EDGE_T": edge_t, "SPEC_Y": SPEC_Y, "SPEC_C": SPEC_C, "DARK_Y": DARK_Y}


def luminance(frame):
    return [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in frame.rgb]


def gradient(frame, lum=None):
    """Per-pixel gradient magnitude; None on the border (no central difference there)."""
    w, h = frame.width, frame.height
    lum = lum if lum is not None else luminance(frame)
    g = [None] * (w * h)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            i = y * w + x
            gx = (lum[i + 1] - lum[i - 1]) / 2.0
            gy = (lum[i + w] - lum[i - w]) / 2.0
            g[i] = (gx * gx + gy * gy) ** 0.5
    return g


class Field:
    """Precomputed luminance and gradient for one decoded frame."""

    def __init__(self, frame, edge_t=EDGE_T):
        self.frame = frame
        self.edge_t = edge_t
        self.lum = luminance(frame) if frame.decoded else []
        self.grad = gradient(frame, self.lum) if frame.decoded else []

    def frame_edge_fraction(self):
        vals = [v for v in self.grad if v is not None]
        return sum(v >= self.edge_t for v in vals) / len(vals) if vals else None


def region_reading(field, region, backlit=False):
    fr = field.frame
    names = ("luminance_gradient", "edge_strength", "specular", "diffuse", "transmission_r_over_b")
    if not fr.decoded:
        return {n: UNMEASURED for n in names}
    x0, y0, x1, y1 = region
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(fr.width, x1), min(fr.height, y1)
    idx = [y * fr.width + x for y in range(y0, y1) for x in range(x0, x1)]
    if not idx:
        return {n: UNMEASURED for n in names}
    grads = [field.grad[i] for i in idx if field.grad[i] is not None]
    lum = [field.lum[i] for i in idx]
    px = [fr.rgb[i] for i in idx]
    spec = sum(1 for i, (r, g, b) in zip(idx, px)
               if field.lum[i] >= SPEC_Y and max(r, g, b) - min(r, g, b) <= SPEC_C)
    diff = sum(1 for v in lum if DARK_Y < v < SPEC_Y)
    out = {
        "luminance_gradient": sum(grads) / len(grads) if grads else None,
        "edge_strength": sum(v >= field.edge_t for v in grads) / len(grads) if grads else None,
        "specular": spec / len(idx),
        "diffuse": diff / len(idx),
        "pixels": len(idx),
    }
    if backlit:
        mr = sum(p[0] for p in px) / len(px)
        mb = sum(p[2] for p in px) / len(px)
        out["transmission_r_over_b"] = (mr / mb) if mb else None
    else:
        out["transmission_r_over_b"] = NOT_APPLICABLE
    return out
