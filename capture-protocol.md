# Capture Protocol

The repo's largest blocker is not analysis. It is that `ThicknessReading.
measurable` returns `False` for every frame on record, because thickness and
boundary sharpness need **raking light** and no frame has it.

Everything in tier 2 — the thickness map, boundary sharpness, concentration —
is waiting on five photographs and one lamp.

---

## The five positions

| # | position | hand | camera | light | resolves |
|---|---|---|---|---|---|
| 1 | **palm flat** | palm up, flat as a table; fingers spread moderately — not stretched, not closed; thumb abducted ~45° | directly above, 12–18 in | single source, side, skimming | crease map, zone coverage, gross girth |
| 2 | **fingertips** | relaxed, palm up, fingers slightly curled as if holding a softball | close, 6–8 in, focused on the pads | side, skimming across the pads | pad texture, proximal phalanx pad markers |
| 3 | **thenar** | thumb extended, palm rotated ~30° toward camera, thenar facing the light | from the thumb side, close | raking across the thenar mass | thenar volume, thumb crotch, thumb pad |
| 4 | **dorsal** | back of hand up, flat; fingers together or slightly spread | directly above | side, same angle as position 1 | dorsal event marks, vein and tendon relief, MCP state |
| 5 | **lateral raking** | flat, palm up | from the wrist side, ~30° off vertical | opposite side — camera at 6 o'clock, light at 12 | **thickness map**, boundary sharpness, hypothenar relief |

**Position 5 is the one that unblocks tier 2.** Without it the state readout
stays ambiguous no matter how many other frames exist.
`CaptureSession.resolves_tier_2` checks for exactly that position in raking
light, and nothing else satisfies it.

---

## The five rules

| | rule | why |
|---|---|---|
| 1 | **Both hands, all five positions, same session** | bilateral comparison is its own control. Asymmetric load distribution — one hand loading precision grip, the other loading a stabilizer — is readable only across the pair. |
| 2 | **Same light geometry every frame** | comparison across time requires it. A state that changes because the lamp moved is not a finding. |
| 3 | **Shoot before and after any dirty work** | the paired grime control: same hands, one variable. This is the demonstration in `known_failure_cases.md`, produced on purpose instead of by accident. |
| 4 | **Shoot on a schedule, not when something happens** | the narrative gate. Disposition-gated capture samples the anomaly and hides the maintained baseline — the opposite of what the band readout needs. |
| 5 | **Keep a dated log** | "Monday, post-shower, 70 hr driving week, yard work Saturday" is enough. The map integrates a load history; without the history the image is uninterpretable. |

`CaptureSession.problems` reports every one of these that a session misses,
rather than accepting the frames and discovering later that they cannot be read.

### The trigger, specifically

```
current gate    notice → decide → photograph      ← depends on disposition
replaced by     fuel stop · odometer crossing a round number ·
                week rollover · log entry
```

`Trigger.NOTICED_SOMETHING` exists in the enum and `is_scheduled` returns
`False` for it. The failure mode is named rather than omitted, so a session
built on it says so in its own report.

**The boring frames are the expensive ones.** No incentive in any population
produces them, which is why the fix has to be a trigger and not a resolution.

---

## Equipment: one light is the specification, not a compromise

Worth stating plainly, because the intuition runs the wrong way.

**A single hard source is the requirement.** Two lights, a softbox, a bounce, or
ordinary room light all produce *fill* — and fill is what erases the shadow that
carries the thickness information. The measurement lives in the terminator
between a raised zone and a thin one, and fill light climbs into it and flattens
it out.

So one bare LED is the correct instrument. What matters is not how many lights
there are but:

| | requirement | why |
|---|---|---|
| **kill the ambient** | shoot in a dark room, that lamp only | any second source is fill, and fill destroys the reading |
| **get it LOW** | near the plane of the hand, not above it | height is the single most common failure. Light from above grazes *along* the surface; the map needs it crossing *across* |
| **flat surface, not vertical** | hand on a table, not pressed to a wall | pressing distorts the tissue being measured |
| **camera perpendicular** | to the hand's plane, not to the light | position 5 puts them opposite each other on purpose |
| **brace the phone** | table edge, book, anything | motion blur has cost more detail in this series than light angle has |

Every one of those is free. None of them is a hardware problem, and a better
camera fixes none of them.

The specimen series has failed on **height and blur**, repeatedly — not on
equipment.

---

## What this unblocks, in order

1. **Tier 2 becomes measurable at all.** Currently every state readout in this
   repo returns `unresolved` or `NOT MEASURABLE`.
2. **The band thresholds stop being stipulated.** `ContrastThresholds` ships
   with `is_evidence_based` returning `False` for every value. A dated series in
   consistent light is what replaces them.
3. **The calibration standard gets its first entries.** Load history, dated
   band-state series, maintenance practice — the fields in
   `calibration-standard.md` that exist only as schemas.
4. **The seasonal prediction becomes testable.** Failure rate peaking at
   transitions rather than at peak load needs a dated series across a shoulder
   season, and nothing else.

---

## Status

Not run. The protocol is five photographs, one lamp, and a notebook line, and it
is the cheapest thing standing between this repo and its own tier-2 metrics.

Everything specified here is geometry and cadence. None of it is stipulated in
the sense the rest of the repo uses that word — there are no thresholds to
falsify, only a procedure to follow or not follow.
