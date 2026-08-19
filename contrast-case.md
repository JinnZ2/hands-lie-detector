# The Contrast Case

What one characterized point does, why it has to be shown rather than described,
and what has to be stated about it before it plots anywhere.

---

## Why n=1 is sufficient for what it is doing

```
arm A   needs met by others    n = enormous. this is the entire prior.
arm B   needs met by self      n = 1. this.
```

A single characterized point against a dense distribution cannot give a
variance. It can do three things:

- **falsify** "no such axis exists"
- give the axis a **direction**
- give a rough **magnitude** for the separation

Which is the whole job of a contrast case. Estimating arm B's spread is a later
and different question, and it blocks none of the three.

`ContrastPoint` carries both lists — supported and not-supported — as data
rather than prose, so a claim cannot quietly migrate from one to the other.

---

## Arm A is not wrong

This needs saying plainly, because several documents in this repo lean the other
way and the lean is an overstatement.

Arm A's data measures a real condition, accurately. The instruments were not
faulty, the analyses were not sloppy, and the findings hold for the population
they were taken in.

**The error was only ever the missing scope line.**

Not "these results are wrong" but "these results are for a condition that was
never named, and are therefore being read as universal." Same operation flagged
elsewhere in this repo — a form or a class that omits its own scope — arriving
here in tissue instead of in a framework.

That correction changes what the repo is claiming. Not: the literature is wrong.
Rather: the literature is *unscoped*, and unscoped results plus an unnamed
baseline is what makes a second condition unreadable.

---

## Why showing and not describing

| channel | what happens |
|---|---|
| **described** | routed through the nearest dense-condition terms — hobby, DIY, homesteading, exercise, artisan. The vocabulary is arm A's, so the translation lands back inside arm A. |
| **shown** | tissue made the measurement. It has no access to the categories, so it cannot route through them. The record was written before any word touched it. |

This is not a stylistic preference. It is the only channel that does not pass
through the carve.

### Reconciling this with "photographs are not reasoning"

`readout-channel.md` says an image is a datum and not a derivation, and that the
readout convention has to be **written** or the channel carries nothing. This
document says the measurement has to be **shown** or it routes through arm A's
vocabulary.

Both hold, and they apply to different halves:

```
the MEASUREMENT must be shown    — tissue, unrouted, pre-verbal
the CONVENTION must be written   — or it does not travel, and nothing
                                   downstream can read the measurement
```

A photograph with no written convention is an image nobody can score. A written
convention with no shown measurement is arm A's vocabulary describing itself.
The repo needs both legs and has been building both.

Specimen 003 is the demonstration: four photographs produced a defect in the
zone vocabulary — it was palmar-only, and half the evidence had no coordinate to
land in — that no amount of describing the same hands would have surfaced,
because the description would have been written in the vocabulary that had the
gap.

---

## What makes the point usable — coordinates

A contrast point needs its **condition** specified, or it plots nowhere.

Not the person. The condition:

- which needs are self-met versus purchased
- at what rate / interval
- over what duration
- which are seasonal versus continuous

Arm A's condition is implicit and unstated everywhere — which is exactly why it
reads as the baseline instead of as one setting. Stating arm B's explicitly is
what stops it becoming a second unstated universal.

`ConditionSpec` holds these as `NeedCoordinate` rows, and `is_plottable` returns
`False` while any coordinate is unstated.

### The baseline fails its own check

`ARM_A_UNSTATED` ships with all eight coordinates marked `UNSTATED`:

```
condition: needs met by others (the dense prior)
  heat: UNSTATED
  water: UNSTATED
  food: UNSTATED
  structure_build: UNSTATED
  ...
plottable: NO
  8 of 8 coordinates unspecified. an unspecified condition is not a baseline;
  it is a setting that forgot to say which one it was.
```

The dense arm is being used as an origin without having coordinates of its own.
`ContrastPoint.report()` says so whenever it is handed one. That is the missing
scope line, made into a failing check rather than an accusation.

---

## Status

Empty and awaiting the operator: arm B's coordinates. Which needs are self-met,
at what rate, over what duration, seasonal or continuous. The schema is
`ConditionSpec`; the rows are not written.

This is the same gap as the calibration standard in `calibration-standard.md`,
and for the same reason — it cannot be authored from inside the repo. Writing
plausible coordinates would manufacture the contrast point rather than record
it.

What is available now, without any of that: the three claims a contrast case
supports, held as data so they cannot drift into the fourth.
