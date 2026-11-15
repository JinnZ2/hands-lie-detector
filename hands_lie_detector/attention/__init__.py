"""
Attention-as-sensor: an observer's attention, read as a probe and checked
against the physics of the frame. PROPOSED, co-authored.

See `attention-as-sensor.md`. Pipeline:

    frame (PNG / PPM, or coordinates only)
      -> attention logs, one per observer, taken before any task prompt; never merged
      -> physics per region: luminance gradient, edge strength, specular / diffuse,
         transmission (backlit frames only: path opacity)
      -> per observer: landings vs light-mass boundaries, with random-region and
         shuffled-order controls
      -> per cell: CALIBRATED_POINTER / BIAS / BLIND_SPOT / NEITHER
      -> across observers: which cells each attended (the divergence map)

Nothing in this package interprets a reading. Bounds travel with every result.
"""

from . import image, log, maps, physics
from .run import run

__all__ = ["image", "log", "maps", "physics", "run"]
