# Backlit Transmission — the path-opacity map

Status: **PROPOSED**, co-authored. Not run on any frame. No threshold is
measured; every confound beyond the ones in the delivered readout is a
candidate to check, not a finding.

The readout below is kept as delivered. The sections after it are the
repo's reading of it against what is already here.

---

## READOUT: BACKLIT TRANSMISSION (thickness map) — as delivered

```
conditions: light source behind hand, camera facing palm
  glow (red/orange)   → thin tissue: margins, finger webs
  translucent         → moderate
  opaque / dark       → dense: palm center, finger-base pads
confounds: bone, muscle, callus, fat all read as "opaque"
  → opacity = thickness, NOT composition
pair with: side-lit relief photo (texture, callus edge)
  → transmission says how thick; relief says what kind
log: light angle, source, time, pre/post wash
status: PROPOSED
```

---

## The conflict this resolves — two quantities called "thickness"

`band-not-scale.md` (instrument limit) and `band/contrast.py` say a backlit
frame **cannot** resolve thickness: `LightCondition.BACKLIT.resolves_thickness`
is `False`, and only `RAKING` passes. This readout says a backlit frame **gives**
a thickness map. Both are right, because they name two different quantities.

| | raking light (position 5) | backlit transmission (this readout) |
|---|---|---|
| what light does | skims the surface, casts shadow at relief edges | passes through the whole hand |
| quantity | **plate thickness**: epidermal / callus depth, boundary sharpness | **path opacity**: total attenuation along the light path |
| what contributes | the outer layer only | everything on the path: bone, muscle, tendon, fat, blood, callus |
| scale of the signal | millimetres (callus edge) | centimetres (margin vs palm centre) |
| feeds | tier 2 (thickness map, boundary, concentration) | nothing in tier 2 |

So the existing statement stands for the quantity it is about. Backlit frames
cannot resolve the callus plate map, and this readout does not change
`LightCondition.BACKLIT.resolves_thickness`. What transmission adds is a
separate, gross readout. To keep the two from merging, this repo calls it
**path opacity**, not thickness, wherever the two could be read together.

**DERIVED.** Callus is a thin layer on a thick path. Through the palm centre
the light crosses metacarpals, interossei and the thenar or hypothenar masses
as well as the plate. The plate's share of total opacity is small, so a
change in callus is unlikely to show in the transmission map at all. This is
the delivered confound line taken one step on: opacity is not composition,
and it is also dominated by the layers this repo does not read. The size of
the plate's share is not measured; that it is small is an inference from
relative layer depth.

## Pairing — already in the protocol

"Pair with side-lit relief" is position 5 (lateral raking) in
`capture-protocol.md`. That is the one frame that resolves tier 2.
Transmission says how much is on the path. Relief says where the plate edges
are. The two do not share a failure mode:
- transmission fails by exposure and perfusion;
- relief fails by angle and occlusion.

That makes them a split pair in the `channel-split.md` sense, two
instruments rather than one reading taken twice.

## Confounds to check, beyond the delivered list (PROPOSED)

- **Blood.** The red/orange glow is light that survived haemoglobin
  absorption, so perfusion changes the map. A warm wash dilates vessels; cold
  constricts them. That is why the delivered log carries pre/post wash and
  time. Add hand temperature, or time since wash, as a number.
- **Exposure.** Auto-exposure and auto-gain rescale the whole frame, so
  "brighter" can mean "camera compensated". Lock exposure, or carry a
  known-neutral patch, per the exposure-reference rule in
  `capture-protocol.md`.
- **Source.** Spectrum, intensity and distance to the hand. Log the source
  model and distance, not only "source".
- **Contact.** Pressing the hand against the source compresses tissue and
  blanches it, which changes both terms above.

## Rule zero extends here — vein pattern

**DERIVED, check before any frame leaves the device.** Rule zero
(`capture-protocol.md`) treats a close raking shot of a fingertip as a
fingerprint capture. A backlit palm has the same problem with a different
pattern: transmission shows the vein network, and palm-vein recognition is a
biometric modality.

The ridge blur in rule zero does not cover this. Ridge pitch is ~0.45 mm,
while veins and the load structure both sit at millimetres, so a
ridge-pitch low-pass leaves the veins intact.

What still separates them is the scale of this readout's own signal.
Margin-versus-centre opacity is a centimetre-scale gradient, so a blur or
downsample at roughly centimetre scale keeps the path-opacity map and
removes the vein detail. Until that is shown on a frame, treat every backlit
frame as biometric:
- keep it on the device;
- strip EXIF;
- do not send it to a model unblurred.

## Log fields

Delivered:
- light angle
- source
- time
- pre/post wash

Added (PROPOSED):
- exposure locked (yes/no) or neutral patch present
- source distance
- hand temperature or minutes since wash
- whether the hand touched the source
- blur scale applied before transmission

## What this readout cannot say

- Composition. Bone, muscle, callus and fat read alike (delivered).
- Callus plate thickness, boundary sharpness or concentration (the tier-2
  items). Those still need position 5.
- Anything distributional. One hand, one session, per `readout-channel.md`.
