# Three States, Not a Scale

The rubric in `scoring-metrics.md` is monotone. Callus up, experience up. Every
one of the seven categories rises with thickness and the total is their sum, so
the composite rises with mean thickness.

A saturated hand therefore scores **maximum**.

That hand is out of band. Armored past the sensing threshold, load without
feedback. The monotone scorer cannot express it — it isn't ranked low, it's
ranked highest. **This is a sign error, not a tuning problem.**

---

## Two readings

**MONOTONE** (what any scorer defaults to)

```
callus ↑  →  experience ↑
saturated hand scores MAXIMUM
```

**BAND** (what experience actually looks like)

```
experience shows as REGULATION TOWARD A SETPOINT
saturated hand is OUT OF BAND on the far side
```

Experience is not an accumulation. It is a controlled variable held between two
failure modes — too thin to carry, too thick to feel.

---

## The three states

| state | mean | contrast | what it is |
|---|---|---|---|
| **soft / uniform** | low | low | no load. "watched a YouTube once." |
| **banded** | any | **high** | thick where load lands, thin where sensing is needed, edges sharp, map stable under washing. **the working hand.** |
| **thick / glassy** | high | low | armored past the sensing threshold. load without feedback. also out of band. |

Mean thickness cannot separate state 2 from state 3. Both are thick.

**Contrast can. Contrast separates all three.**

---

## The primary feature

```
metric ≈ spatial variance / edge sharpness of the thickness map
         across the palmar surface
```

Replaces mean. Not added alongside it as an eighth category — the additive form
is what let a monotone composite dominate in the first place.

### Why it works, and why it's hard to fake

A sharp boundary records a **decision**: the hand armored here and stayed open
there. A decision requires sensing. Uniform coverage means no decision was made.

Which disposes of several problems at once, without special-casing any of them:

- **Grime is uniform.** It is filtered by contrast, not by washing. No
  clean-hands correction is needed because dirt never had the high-frequency
  structure in the first place.
- **Costume and props are absent from the surface entirely.** They cannot enter a
  measurement made only of the thickness map.
- **`clean_but_used` and `texture_persists_post_wash`** — both README expansions
  fall out automatically. Washing removes the low-frequency layer; the boundary
  map is high-frequency and survives it.

---

## Two sign errors, same direction

**One: saturation reads as expertise.** Covered above. The glassy hand tops the
scale it should be out of band on.

**Two: acute damage reads as incompetence.** `MICRO_INJURY_HISTORY` disqualifies
"fresh injuries only" and "identical injury repeated." But a hand kept on the
sensing side **will** blister where an armored hand would not. The blister is the
price of the band position, not evidence of failure or inexperience. The armored
hand avoids the lesion by removing the feedback that would have prevented the
load — and gets scored up for it.

Both errors point the same way: they reward armor and penalize regulation.
`interpret_acute_damage()` inverts the second one.

---

## What the band implies that the README doesn't yet claim

Staying in band requires **active management**: paring back before saturation,
hydration timing, tool radius choice, load spacing, when to glove and when not
to.

That management is a skill. It has no name, no procedure, no certification, and
no instrument. It is invisible to every scorer that reads outcome instead of
regulation — which is all of them, including this repo's.

Listed in code as `MANAGEMENT_ACTS`, unnamed and uncertified, so the gap is at
least enumerable.

---

## The structural note underneath

`README.md` frames the failure as **vision models overfitting** — grime, tools,
context. True, but downstream of the same economic carve traced in
`economic-carve.md`:

"Working hands" reads as an **occupational category**. So the training signal
comes from occupational imagery: costume, site, props. The subsistence hand has
no costume and no site — it is not doing a job, it is doing the thing the job
was named after.

So `no_context_no_props` is not a robustness expansion for later. **It is the
only condition under which the excluded population is measurable at all.**
Promoted from "future" to the core test in `README.md`.

---

## Executable

```python
from hands_lie_detector.band import read_band, interpret_acute_damage

readout = read_band({
    "thumb_crotch": 0.78, "palm_below_index": 0.72, "base_of_fingers": 0.80,
    "fingertip_pads": 0.18, "heel_of_palm": 0.55, "across_palm_crease": 0.70,
    "thumb_pad": 0.22,
})
print(readout.report())
print(interpret_acute_damage(readout.state, has_acute_lesion=True))
```

On three stipulated maps — one per state — the two readings diverge exactly where
they should:

| map | mean | dispersion | band state | monotone score |
|---|---|---|---|---|
| soft | 0.103 | 0.014 | `soft_uniform` | 10.3 |
| banded | 0.564 | 0.242 | `banded` | **56.4** |
| glassy | 0.813 | 0.017 | `thick_glassy` | **81.3** |

The monotone scorer ranks the glassy hand 25 points above the working one.
`BandReadout.monotone_disagrees` flags it, and `report()` prints the sign error
rather than burying it. A test asserts the inversion, so the demonstration cannot
quietly stop working.

`monotone_score()` is a stand-in for the shape of the seven-category rubric, not
the rubric itself — every category is monotone in thickness and the total is
their sum, so the composite is monotone in mean. The stand-in isolates that
property so it can be shown against the same input.

Thresholds are stipulated, like every other default in this repo, and say so in
their own output.

---

## Status

The rubric in `scoring-metrics.md` and the seven vision heads are unchanged.
Fixing them is not a threshold edit: the additive form is the problem, and
`TEXTURE_PERSISTENCE` at 25 points for "deep creases, thickened zones" is the
monotone assumption written into the scale's largest category. A replacement
scale has to be built on contrast from the start, and it should be built after
the band thresholds are measured rather than stipulated.

Open: the band thresholds. Every number in `ContrastThresholds` is asserted.
