# The Wear Taxonomy Was Already There

Tribology classifies wear by **mechanism, never by application**. Nobody sorts
wear by industry — that is the entire point of the taxonomy, and it is
domain-blind because it had to be.

Which means this repo has been rebuilding, by hand and badly, a spine that
already exists.

| mode | mechanism | tissue reading |
|---|---|---|
| **adhesive** | surfaces grip; the junction shears below the interface | friction-peak delamination — the blister roof shears *below* the gripping surface, not at it |
| **abrasive** | hard particle or rough counterface ploughs | bark, grit, clay, sand |
| **fatigue** | subsurface cyclic damage, failure at depth | callus is fatigue-driven remodeling; blister is fatigue delamination |
| **corrosive** | chemical attack of the surface layer | alkali, solvent, defatting — the scrub |
| **fretting** | small-amplitude oscillation at a fixed contact | a ring band. textbook fretting geometry |

Reading a worn component backward to its service condition is routine failure
analysis. Running it on tissue needs no new vocabulary. `LOAD_MODE_TO_WEAR` maps
this repo's ad-hoc `LoadMode` enum onto the standard one.

---

## Transfer 1 — wear is a system property, not a material property

```
load · velocity · geometry · COUNTERFACE · lubricant · cycles
```

So **the hand alone is an incomplete specimen.** The tool carries the conjugate
record. That is not a convenience shortcut, it is required by the formalism:
handle wear and palm wear are one measurement taken from two sides.

`WearSystem.is_complete_specimen` returns `False` without a counterface, and says
so in its report. Photographing the handle is not optional extra data — it is the
other half of the measurement the repo has been taking one-sided.

## Transfer 2 — run-in

Surfaces wear fast, conform, then settle to a low steady rate. **The conformed
state is optimal, and it is maintained rather than final.**

The band is run-in held:

| run-in state | band |
|---|---|
| pre-run-in | `soft` — no conformity yet |
| conformed | the band — low wear rate, functional |
| carried past function | `uniform_thick` at saturation — the surface still exists, the interface no longer works |

### Where the transfer breaks — the interesting part

Steel only degrades. **Skin remodels toward the load.** A negative feedback loop,
and there is no complete engineering analogue: tribofilms and work-hardening are
partial at best.

That residual is the part with no literature, and it is where this repo's actual
subject lives. `RESIDUAL_WITHOUT_ANALOGUE` holds the statement in code so it
travels with the module.

---

## It settles the category question

A wear scar's morphology identifies the **mechanism**. The mechanism does not
know the **application**. The application is where the codes live.

Own land versus someone else's land: same load, same counterface, same cycles →
identical scar. The category has no term in the wear equation to modify.

Not a claim that categories are unimportant. A claim that they are **downstream
of a measurement the material already took, before anyone classified it.**

### The natural experiment this makes available

Same tractor, same loader, same terrain class, same hands, within one operator
and a two-year window:

- **own land** → no code, no wage, "hobby"
- **other people's land** → coded, waged, "operator"

Identical mechanics. Category flipped. Not constructed — it happened.

Prediction: **flat.** No discontinuity in deposit, in band position, or in
anything measurable. Only the classification moves. That is the cleanest
available test of the claim in `economic-carve.md` that the economic code
carries zero physical information, and it is the discontinuity test with the
confounds removed by circumstance rather than by design.

---

## Two surfaces, two instruments, two clocks

The repo specced only the palmar one.

| | palmar | dorsal |
|---|---|---|
| tissue | thick, adaptive | thin, mobile, over metacarpals |
| callus | possible | **impossible** — no adaptation route |
| dominant mode | adhesive / fatigue | impact, snag, edge |
| what it does | **integrates** over weeks | **marks** a single moment |
| what it records | load history | an event |
| clock | keratin turnover, 2–4 wk | healing, ~2 wk |

A dorsal count says nothing about accumulated load and everything about
**conditions**. So it gets its own track: `integration.event_log`, where
`EventLog.carries_load_history` returns `False` unconditionally and the module
declines to emit a load estimate. `DorsalMark` raises on a palmar zone, and
`ThicknessReading` raises on a dorsal one — the two instruments refuse each
other's inputs.

### The same two channels on the other end of the body

The palm's pair — material removal versus delamination — is the sole's pair too:

| | palm | sole |
|---|---|---|
| removal channel | abrasion of the stratum corneum | lug height loss |
| separation channel | blister: fatigue delamination | crack initiation at the flex line |
| driver of the fast one | shear cycles at a fixed contact | flex cycles at the same line |

Which is why a boot can reach structural failure in four months without high
mileage: the fast channel is fatigue, not abrasion, and solvent exposure plus
cold cycling lowers its initiation threshold. See `sole-audit.md` — and note
that this makes time-to-failure useless as a distance proxy, which
`SoleReading.time_to_failure_supports_distance_claim` enforces.

### Correction — the dorsum has one adaptation route after all

An earlier version of this document, and `EventLog.carries_load_history`,
returned `False` unconditionally: dorsal tissue has no adaptation route, so it
records events and never history.

That is too strong, and a field discriminator in routine use exposes it.

The dorsum is not a **grip** contact surface, so grip and shear load deposit
nothing there — that part holds, and it is the strong claim. But the dorsum
**is** a contact surface for striking, and repeated axial impact at a fixed
knuckle does remodel it. So the surface runs two channels:

| channel | deposits? | clock |
|---|---|---|
| event log — laceration, abrasion, contusion, split | no, heals away | ~2 weeks |
| repeated impact at a fixed site — soft-tissue thickening | **yes** | persists |

`carries_grip_load_history` returns `False` unconditionally, as before.
`carries_impact_history` is the new, narrower channel.

### The discriminator: count and concentration on the dorsum

Two histories that both live on the MCP row and are not the same thing.

| | **edge-strike field** (mechanic) | **repeated impact** (bare-knuckle striker) |
|---|---|---|
| count | hundreds | few |
| distribution | scattered across the dorsum and MCP row, varied sites | concentrated on the 2nd and 3rd MCP heads — the knuckles that land |
| depth | superficial, skin-level | structural: soft-tissue thickening at the strike points |
| mechanism | laceration and abrasion against an edge, after a sudden release | repeated axial impact |
| what it records | event **count** | one geometry, repeated |
| remodeling | none — scar only | present |

The mechanism behind the mechanic signature: the hand is in a confined volume
with edges in it, a fastener releases suddenly, and the knuckle travels. **Every
reach has a different geometry**, so every mark lands somewhere new. High count,
low concentration, no remodeling.

The striker's is the inverse — few events, one geometry, and the only dorsal case
that deposits.

Which is the **concentration axis** from `band-not-scale.md`, running on the
dorsal surface. Same quantity, different tissue: it reads how varied the contact
geometry was, and — as on the palm — it says nothing about competence in either
case.

`dorsal_signature()` implements it. Thresholds are stipulated from one
operator's field use, not fitted: this is `TESTIMONY`, but testimony from a dense
observational sample rather than from a single case, which is more than most of
the tables in this repo have behind them.

### Why dorsal marks cluster in cold work

```
cold hand  →  tactile feedback down
              fine motor down
              skin elasticity down → splits rather than deforms

cold work  →  edges. chains, hooks, latches, frozen fittings:
              things that resist and then release suddenly
```

The injury is not from more force. It is **reduced sensing plus higher edge
density**.

Which is the band again, driven from outside: cold moves the hand toward the
low-sensing end of the window **with no change in callus at all**. Environmental
de-calibration. Held at roughly 55% that de-calibration is the dominant term
against simple edge exposure — a stated split, not a settled one.

---

## Deposit and draw — the third functional-form error

| class | signature | effect |
|---|---|---|
| **deposit** | large force, large contact, cyclic | builds plate. consumes tissue. **writes the map** |
| **draw / spend** | near-zero force, small radius, isometric | deposits nothing. **spends** the sensing capacity the deposit classes produced |
| **draw / suppress** | sustained high-frequency input | knocks the sensing channel down **directly**, no callus change, recovers on its own timescale |

An hours-weighted model puts all three on one axis as parallel inputs.
Physically they are on opposite sides of the ledger: one is a load, the others
are demands on the capacity that load produced.

**No coefficient expresses a sign**, any more than a sum expresses a gate. This
is the third instance in the same stack — weights, gates, and now sign — and it
belongs in the index in `gated-not-summed.md`.

### Decoupling

Some domains do both at once. An open-station tractor: large-diameter wheel, high
effort, uneven ground, constant corrective input at fixed geometry — plus
hand-arm vibration through the wheel and whole-body through the seat. It
**builds plate while knocking sensing down**.

So during that work, plate map and band position **decouple**, and neither one
predicts the other. `DomainSignature.decouples_map_from_band` flags it.

Everything else in the series was one or the other:

| domain | class |
|---|---|
| rotary / clamp | deposit only |
| probe work | draw only |
| chainsaw | suppress only |
| **open-station tractor** | **deposit + suppress** |

The tractor is the first: large-diameter wheel at high effort over uneven ground
gives constant corrective input at fixed geometry and high cycle count; loader
levers add moderate force at sustained wrap grip; and the open station puts
hand-arm vibration through the wheel and whole-body through the seat. Plate goes
up while sensing goes down, in the same hour.

### One word, four contact distributions

"Firewood" is not a domain. It is a bundle:

| sub-domain | modes |
|---|---|
| cut | vibration, two-hand static, weight held out |
| split | impact, axial, palm-heel + wrap, high peak / low cycle |
| stack | variable grip, bark abrasion, point loads, lift-carry-place |
| base | pallets, mat, leveling — awkward-object handling |

The deposit from the word is the **sum of four contact distributions**, which is
exactly why it cannot resolve to a single site — and why it produces the diffuse,
low-concentration map that broke the contrast metric.
`DomainSignature.is_bundle` flags it, and the four components ship as their own
signatures so the bundle can be decomposed rather than averaged.

---

## Status

Stipulated, as always: every `load_classes` assignment in `DEFAULT_DOMAINS`, the
`LOAD_MODE_TO_WEAR` mapping, and the 55% split on the cold mechanism.

Not run: the natural experiment. It needs a tissue record spanning the own-land
/ other-people's-land boundary, which means the dated band series in
`calibration-standard.md` — the same blocking gap as everywhere else.

Not collected: any counterface. Every wear measurement in this repo is half a
specimen until tool handles are photographed alongside hands.
