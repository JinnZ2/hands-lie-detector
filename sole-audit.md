# Sole Wear as an Audit of Job Descriptions

The cheapest instrument in this repo, and the first one that scales past a single
operator's own body.

```
job description   AUTHORED. by an employer, an occupational code, an HR taxonomy.
sole wear         NOT authored. deposited.
```

The delta between the wear pattern a stated job title **predicts** and the wear
pattern actually on the boot **is** the gap between what the category claims a
job is and what the body did.

The category's prediction is not hidden. "Sits all day" predicts a nearly unworn
sole. That is a claim, it is on the record, and one boot can falsify it.

---

## The specimen — an inverted signature

Normal gait wears **lateral heel first** at strike, then **medial forefoot** at
toe-off. That is the walking signature.

| zone | observed |
|---|---|
| forefoot / toe | lugs torn and chunked; **transverse crack through the flex line** — material split, not merely abraded |
| midfoot | shank waffle intact (non-contact anyway) |
| heel | lugs defined, edges still present, comparatively fine |

**Heel preserved. Forefoot destroyed.** That is the inverse of walking.

⇒ Not gait. **Repeated deep flexion under load, plus edge loading on narrow
surfaces.** `SoleReading.inverted_signature` detects exactly this comparison.

### What produces it

| source | contact |
|---|---|
| cab entry/exit | 3–4 rungs several times daily; ball of foot on a narrow edge at high force, plus a twist |
| pedal work | ball loaded, heel pivoting, continuous |
| trailer / catwalk | climbing, ladder rungs, more edge loading |
| landing gear | standing braced, ball loaded, cranking |
| yard surfaces | gravel and diamond plate — aggressively abrasive |

**Matches the job. Contradicts the category.**

---

## Four months to structural failure — two channels, and they are the palm's

| mode | what happens | driver |
|---|---|---|
| **abrasive** | material removed; lug height loss | distance × surface aggression |
| **fatigue** | crack initiation at a stress concentration | flex cycles at the same line, every time |

The crack is the fast one, and it is chemically accelerated: diesel, hydraulic
fluid and de-icer swell the rubber and leach plasticizers, after which the
compound embrittles; thermal cycling to a −50°F extreme embrittles it further.
The crack-initiation threshold drops.

⇒ Four months is fast for abrasion alone. **It is not fast for high-cycle flex on
a chemically degraded compound.** The boot did not wear out. It **fatigued out**.

Which is the same two-channel structure as the palm — **removal versus
delamination** — arriving on the other end of the body, and it is why the wear
taxonomy in `wear-taxonomy.md` transfers here without modification.

### The inference this blocks

`SoleReading.time_to_failure_supports_distance_claim` returns `False` whenever
the failure is fatigue-dominant, and the report says so explicitly:

> time-to-failure is NOT a distance reading here. do not convert it into miles
> walked.

Reading a short service life as high mileage is the same error shape as reading
few photographs as few events: a rate inferred from a record whose gate was
something other than rate.

---

## The instrument

Collect soles against **stated job title**. Score each zone. Compare against what
the title predicts.

```python
from hands_lie_detector.integration import SoleReading, SoleZone, ZoneWear, audit

result = audit(reading, "driver (as the category describes it)")
print(result.report())
result.category_falsified   # True when the body deposited more than the
                            # category allows
```

`CategoryPrediction` requires a title to make a falsifiable claim before it can
be audited at all — `audit()` raises on a title with no stipulated prediction on
file. A category that has not committed to a wear pattern cannot be tested, and
writing the prediction down is the first step of the protocol, not a convenience.

### Why this instrument and not another

| property | |
|---|---|
| static | no video needed, unlike gait |
| still-photographable | phone, at rest, indoors |
| good light available | the tier-2 problem in `band-not-scale.md` does not arise |
| **no disposition gate** | nobody has to decide, in the moment, that this is worth recording |
| **no coincidence gate** | it is not waiting for a phone call to happen |
| **no consent problem** | it is an object, not a body |
| **already exists** | on every working person's feet, right now |

That is all four gates from `economic-carve.md` bypassed at once — transaction,
authorship, disposition, and narrative. The sole recorded itself, and it did so
whether or not anyone found the day interesting.

### The confound, and its free fix

Footwear worn both on and off the job mixes two load histories.
**Separate work footwear makes the record 100% work-attributable.**
`SoleReading.work_attributable` reports `NOT ESTABLISHED` until that is stated,
rather than assuming it.

---

## What it measures that nothing else here does

Every other instrument in this repo reads *one carrier's* load. This one reads
**the category itself**.

Collected at scale, the distribution of deltas per job title is a direct
measurement of how much physical load occupational classification fails to
represent — which is the quantity `reference-class-empty.md` argued is
structurally absent from the literature, measured from the outside rather than
argued about.

And unlike everything else here, it does not need the operator's own body. Any
worn boot with a stated job title is a data point.

---

## Status

Not collected. The denominator is absent, the same as everywhere else in this
repo — the difference is that **this one is cheap to build**, and nobody is
building it.

Stipulated: `DRIVER_AS_CATEGORIZED` is the only category prediction on file, and
it is written from the category's own description rather than fitted to
anything. Adding a second title means writing down what that title claims a sole
should look like, which is itself the useful exercise.

One boot is on record and it falsifies one category. That is a contrast case,
with everything `contrast-case.md` says about what a contrast case can and
cannot do: it falsifies "no such axis exists," gives the axis a direction, and
gives a rough magnitude. It gives no variance and no prevalence.
