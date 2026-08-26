# The Knuckle Is a Separate Instrument

The MCP joint is not a grip surface. It is a strike, catch, press and
hyperextension surface, and the palmar readout cannot see it.

The repo's vocabulary was palmar-only until specimen 003, where half the evidence
had nowhere to land. Five dorsal zones were added then. This document is the
instrument that sits on them.

---

## The structural claim

**Palmar load and dorsal load are not correlated.**

A hand can carry intense palmar load with almost no knuckle history — precision
work in a controlled environment. Or intense knuckle history with modest palmar
load — reaching into enclosures, strike-heavy work.

So a model that reads grip zones and infers the rest gets a strike-heavy operator
wrong in a specific direction, and `KnuckleReadout.predicts_palmar_load` returns
`False` unconditionally to keep that inference out of the code.

---

## Three load modes at a joint not built to be a load surface

| mode | mechanism | tissue response | signature of |
|---|---|---|---|
| **direct impact** | knuckle contacts a hard surface — strike, or catch against an enclosure | abrasion, laceration, dorsal scar, pad formation | mechanics, fabricators, builders, fighters |
| **hyperextension** | finger forced backward beyond neutral | volar plate tear, collateral sprain, joint instability | ball sports, falls, tool kickback, heavy lifting |
| **hyperflexion under load** | gripping with the MCP flexed, loaded axially | joint compression, synovitis, capsular thickening | climbers, heavy tool users, drivers |

When the knuckle is a primary load surface, something in the work environment is
hitting the back of the hand — or the hand is hitting something.

---

## Marker taxonomy

| marker | what it is | deposits? |
|---|---|---|
| **knuckle pad** | localized hyperkeratosis over the joint; soft tissue, moves with the skin, no bony component | **yes** |
| **scar** | dorsum broken against something sharp, rough or hot | no — event |
| **carbon stain** | wound open in a carbon-rich environment; carbon bonded to healing tissue | no — event, but permanent and datable |
| **diffuse fullness** | capsular thickening without a discrete pad | **yes** |
| **instability** | extensor or volar apparatus disrupted by sudden force | no — acute in origin |

### Scar location distinguishes the posture

- **MCP scar** → the joint was **flexed** when struck. Hand forward, tight space,
  knuckles leading.
- **Metacarpal shaft scar** → the dorsum was **flat** against a surface, or
  dragged across an edge.

`scar_mechanism()` returns the reading. This is a real discriminator from
location alone, which the palmar map has no equivalent of.

---

## What is not a load marker

Several findings here have clinical differentials a photograph cannot exclude:
rheumatoid nodules, Heberden and Bouchard nodes, gouty tophi.

The separator is a **palpation** finding, not an image finding:

> A knuckle pad is soft tissue — it moves with the skin and has no bony
> component. Heberden and Bouchard nodes are bony and do not move.

So a vision instrument should **report the finding and decline the
differential**. `Differential.requires_clinical_view` returns `True` for every
member.

**This is a load-history instrument, not a diagnostic one.** A knuckle finding
that is painful, progressive, or symmetric without a matching load history is a
clinical question and belongs with a clinician, not with this repo.

---

## The correction this material forced, twice

`event_log.py` has now been corrected in the same direction on two consecutive
passes. Worth keeping both, because the pattern is the point.

| version | claim | why it was wrong |
|---|---|---|
| 1 | `carries_load_history` → `False` unconditionally. Dorsal tissue has no adaptation route. | too strong. Repeated axial impact at a fixed knuckle *does* remodel. |
| 2 | Adaptation route narrowed to **striking**. | still too narrow. Knuckle pads form in carpet layers, tailors and shearers, who strike nothing. |
| 3 | Route is repeated **dorsal contact** — impact, friction and pressure all qualify. | current. |

The claim that survived both corrections is the one about grip: the dorsum is not
a grip surface, so grip and shear deposit nothing there.
`carries_grip_load_history` is still unconditional, and it is the only part of
the original that should have been stated that strongly.

### And it makes the dorsal signature ambiguous

`DorsalSignature.REPEATED_IMPACT` was renamed `REPEATED_CONTACT`, because
concentrated dorsal thickening has at least two routes and the marker alone does
not separate them. **A striker and a carpet layer land in the same class.**

The separator is the co-occurring scar field: strike work carries one, press work
does not. Thickening on its own does not decide it — the same shape as
`UNIFORM_THICK` on the palmar side, ambiguous by construction and saying so.

---

## Falsifiable work predictions

Same form as `sole.CATEGORY_PREDICTIONS`: a marker pattern claims something about
the work, and a carrier can refute it.

| marker pattern | predicted work characteristic |
|---|---|
| pads on 2nd–4th MCP, bilateral | repeated pressing or scraping against flat surfaces — carpet laying, tailoring, machining, grinding |
| pad on the thumb MCP | heavy pinching, tool use, wire work |
| scars on MCP dorsum, varied digits | reaching into enclosures with sharp edges — engine bays, machinery, stock |
| carbon stain in a dorsal scar | hot work or engine work: soot exposure during the open-wound phase |
| hyperextension history | falls, tool kickback, jamming, heavy lifting with the hand forward |
| extensor apparatus disruption | direct impact — striking, or hammering without a guard |
| diffuse MCP fullness, no discrete pad | chronic heavy grip — climbing, heavy tool use, pulling |

All stipulated from mechanism. None fitted.

---

## Why knuckles are invisible to vision models

Not misread. **Unread.**

Training data is overwhelmingly palmar (grip analysis, palmistry, biometric
capture), clean or uniformly dirty, and shot frontally or from overhead — never
raking across the knuckle ridges.

A knuckle pad or dorsal scar is small relative to hand area, low-contrast without
raking light, and **absent from the grip vocabulary the model learned**. When it
does register it tends to be read as "rash", "dermatitis" or "poor skin
condition".

`known_failure_cases.md` documents models reading cleanliness as competence. This
is a worse failure and a different one: there is no category for the marker at
all. The cleanliness case is a **wrong answer**. This one is a **missing
question**.

---

## Capture — three positions the main set does not cover

**Knuckle ridges run transverse across the hand.** Light running down the fingers
grazes along them and resolves nothing. Light crossing them casts shadows into
the valleys. This is why position 4 of `capture-protocol.md` — flat dorsal, side
light — still misses knuckle detail even when the light is low.

| | position | hand | camera | light |
|---|---|---|---|---|
| **A** | dorsal survey | flat, dorsum up, fingers together or slightly spread | directly above | **raking from the fingertip direction toward the wrist**, or from the thumb side |
| **B** | loose fist profile | fingers curled ~90° at the MCP, IP joints extended | dorsal surface facing camera, slightly angled | from the side, skimming across the MCP ridge line |
| **C** | MCP side profile | lateral view, MCP joint centred | from the side | from above or below, grazing the joint |

B is where a scarred knuckle field shows best — in flexion the knuckle becomes a
ridge and any pad, scar or irregularity stands in relief.

**Log per session:** which MCP joints carry markers · bilateral or asymmetric ·
fresh / healing / healed / scarred · pad vs scar vs diffuse swelling · any carbon
staining or embedded material.

---

## Status

Not photographed. The knuckle scar field in the current specimen series is
`TESTIMONY` only — described, never shot under raking light. Positions A and B
would move it to `MEASURED`.

Everything in the marker taxonomy and the prediction table is stipulated from
mechanism. The pad-versus-node differential is the one item here that a
photograph **cannot** settle at any light angle, and the module declines it
rather than guessing.
