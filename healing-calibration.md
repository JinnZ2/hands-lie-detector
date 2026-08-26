# Residual Mark Is Not Proportional to Trauma

```
residual mark  =  trauma  ×  (1 − healing quality)
```

A scar count read straight is monotone in **residual**, and residual is inversely
related to how well the carrier heals. So **a better-healing carrier reads as
less-injured at an identical trauma history.**

This is the **fourth sign error** documented in this scale, and it is the same
shape as the third: a physiological parameter carrying no load information enters
a monotone count uncalibrated, and marks the carrier down for biology.

---

## The four, together

| # | error | the scale rewards | at identical load |
|---|---|---|---|
| 1 | saturation reads as expertise | armor | a hand past the sensing threshold ranks top |
| 2 | acute damage reads as incompetence | absence of lesions | the armored hand avoids the blister by removing the feedback that would have prevented the load |
| 3 | thickness baseline uncalibrated | a higher plate baseline | same history, lower score |
| 4 | **residual mark uncalibrated** | **healing badly** | same trauma, fewer visible marks, lower score |

Three and four are the same operation on different quantities. Both are fixed by
normalizing before the monotone step, and neither is fixed by moving a threshold.

---

## Every factor pushes the same way

Not noise. A fixed direction.

| factor | effect on visible evidence | effect on actual history |
|---|---|---|
| good collagen turnover | scars fade toward invisible | **history is larger than the scars suggest** |
| cleaner remodeling | less hypertrophic scarring | same mechanical events, less residual mark |
| thinner stratum corneum | less visible callus per unit load | same load, less to photograph |
| higher skin elasticity | tissue recovers shape | less permanent creasing from the same deformation |
| better vascular response | bruising and erythema resolve faster | shorter window in which an event is visible at all |
| **hands kept in service** | events overlap rather than stacking into distinct marks | continuous load prevents full recovery between events |

That last row is the one that is not biology. "Healing well because I have to" is a
**functional** constraint: the hand stays in use, so healing is prioritized and no
event fully resolves before the next lands. The marks blur into each other rather
than accumulating as a countable series — which reduces the count while the
history grows.

---

## The seam applies, so read the direction and not the number

Healing rate is a **constitutive parameter for living tissue**. Per the seam in
`economic-carve.md`, the relation transfers across domains and the **coefficient
does not** — those numbers came from sampled human populations.

So `HealingCalibration.residual_factor` is stipulated, `implied_events()` returns
a float with no upper bound on purpose, and the report says to treat the
**direction** as the finding. Ordering is defensible. Magnitude needs calibrating
against the body being read, which is the dated series in
`calibration-standard.md` again.

---

## What no light angle reaches

Breaks and crushes remodel **bone and joint**, not only skin:

- healed fracture callus and bone remodeling
- joint capsule thickening
- ligament laxity from a prior sprain
- tendon adhesion or re-routing after rupture

None of this is visible to surface imaging at any resolution or angle. So a
photographic instrument **systematically under-reads a history containing
fractures**, and it does so in the same direction as everything above.

This is a scope limit, not a resolution problem. `capture-protocol.md` does not
fix it, and neither does a better camera. `BELOW_THE_SURFACE` names the items so
they stay on the record rather than being quietly absent.

---

## The instrument extends past the MCP row

Each joint sits at a different point in the kinematic chain and fails
differently:

| joint | typical load | trauma pattern |
|---|---|---|
| **MCP** | hyperextension, strike, abrasion | sagittal band rupture ("boxer's knuckle"), collateral sprain, knuckle pads — *the only joint of the three that develops an adaptive pad* |
| **PIP** | lateral stress, crush between objects, hyperflexion under load | boutonnière deformity (central slip), collateral ligament tears — *crush injuries land here: the joint sits between two surfaces closing on a hand* |
| **DIP** | pinch trauma, crushing, avulsion | mallet finger (terminal extensor avulsion), jersey finger (FDP avulsion), nail matrix damage |

**DIP trauma writes into the third clock.** The nail matrix sits at that joint, so
a DIP event dates itself by distance from the fold — `KnuckleFinding.
writes_to_nail_clock` flags it and `KnuckleReadout` prompts the cross-check
against `integration/nail.py`.

Which gives one genuinely corroborated channel: a DIP finding and a nail mark at
a consistent age are **two instruments on one event**, not one read twice.

---

## Capability claims are testimony that can fail

A note on the specimen format rather than on tissue.

> "My hands are rough" and "I can hold sub-millimetre placement under near-zero
> force with fingertip feedback only" are both `TESTIMONY`. They are not equally
> strong.

The second names a **performance under stated conditions**. It can be run again
and it can fail. That does not promote it to `MEASURED` — it is still the
carrier's report — but it separates testimony that could be wrong *and would show
it* from testimony that could be wrong quietly.

`SpecimenLine.falsifiable_on_demand` marks the distinction. It matters here
because the sensing test in `band-not-scale.md` is exactly that kind of claim:
sustained precision work at near-zero force, fingertip feedback only, is not
performable with saturated hands.

**And in specimen 010 the demonstration was actually run** — performed in a
separate session with a second model witnessing. `Provenance.DEMONSTRATED` is
the level for that: a capability performed under conditions and seen by someone
other than the carrier.

`Corroboration` records what such a verification is worth. It asks two
independent questions: whether the verification ran **against** the verifier's
documented bias (a hostile-witness argument, which strengthens it) and whether an
**operationalized threshold** existed (which decides whether it reaches
measurement). The second answer is no, so `reaches_measurement` returns `False`.

Hostile-witness credit does not substitute for a stated criterion. Both halves
are true at once: the demonstration is real, and the thing it was judged against
does not exist yet.

---

## Status

Stipulated: every coefficient. `Turnover.HIGH → 0.6`, the two 0.85 multipliers,
all of it. The direction is the defensible part.

Unfixed: the rubric. `MICRO_INJURY_HISTORY` still counts residual marks with no
healing term, so it carries this error live, the same way the thickness
categories carry errors 1 and 3. Four documented sign errors, none fixed in the
shipped scale, for the same reason as always — each is a replacement rather than
an edit.
