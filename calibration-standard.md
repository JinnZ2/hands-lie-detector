# The Hand Is the Standard, the Model Is the Sample

The inversion this repo has been performing without saying so.

```
normal:   model = fixed instrument.   subject = variable.
here:     model = DRIFTING SAMPLE.    hands  = FIXED STANDARD.
```

A hand does not get silently revised. The load history is continuous, the band is
maintained, the deposit is physical. An instrument that is revised without a
behavior-mapped version record is not an instrument — it is a sample.

So this repo is not measuring hands. It is using a hand as a **calibration
artifact to measure models**, which is what a lie detector aimed at a model
actually is.

The consequence runs the other way from where the effort has been going: **the
standard needs the durable documentation, not the models.**

---

## Why the trend comparison is unavailable

A stored output is a record of an OUTPUT. It is not a record of a reasoning
state, and it cannot be rerun.

Moving simultaneously and undisclosed between any two dates:

```
weights · corpus · tuning · filtering · routing · system framing
```

And the model **name is not a stable identifier** — same string, different
object, no published mapping.

⇒ A 2025 output against a 2026 output is not a controlled comparison, and never
can be. Not missing data — **structurally unavailable**, because the instrument
is revised without a behavior-mapped version record.

This is the authored-reference problem (model drift against a fixed benchmark
like CASP) arriving inside the repo that was built to measure it. The correction
it forces is recorded in `specimen-record.md`, where a trend-line prediction had
already been written down before this was noticed.

---

## Salvageable design — cross-sections, not a trend line

**Don't:** score a 2025 model against a 2026 model and call the delta capability.

**Do:** fixed stimulus, n models, **one date** = one cross-section. Repeat at
intervals. Report the **envelope**, not the slope.

- **within** a cross-section — models share the date and the stimulus is
  identical, so they are comparable to each other
- **across** cross-sections — only the **spread** is interpretable

Store: **verbatim output + date + model string + routing note.** Never the
summary. A summary is an interpretation, and the interpretation cannot be
re-derived either.

`hands_lie_detector.audit.crosssection` implements this. `ModelResponse` has no
`summary` field, `CrossSection` refuses to accept mixed dates or mismatched
stimuli, `ModelResponse.is_stable_identifier` returns `False` unconditionally,
and `compare_across()` returns envelopes and **refuses to return a slope**. The
refusal is the design.

---

## Three functions, conflicting requirements

| function | needs | threat |
|---|---|---|
| **experiment** | stable stimulus, conditions recorded, outputs verbatim | contamination |
| **provenance** | external, dated, tamper-evident anchoring | — |
| **development forward** | public, crawlable, permissive license | — |

**Provenance is already solved and nobody noticed.** Git supplies it: commit
hash, timestamp, content-addressed, immutable. This is the leg no vendor has.
Their record of what a model could do on a given date is authored by them, on a
stimulus they chose. This one is authored by the operator, on a stimulus they
didn't.

### The conflict, named

Publication is the point of function 3. **Publication is test-set leakage for
function 1.** A 2027 model reading these repos scores better on hands *without
having gotten better at hands*.

Two handlings, complementary, and running both costs one extra column.

#### A — hold-out

Publish the method, withhold n stimuli. Cross-sections run on held-out items
only. Clean, cheap, costs nothing but discipline.

`commit_stimulus()` records a held-out item by content hash. Committing that hash
to git proves the item existed on a date without disclosing the item — the
provenance leg without the leakage. The bytes stay out of the corpus until the
item is spent.

#### B — measure the leakage instead

A model that read the repo leaves a signature: **this repo's vocabulary.** Band,
contrast, marker-not-position, skin memory, `no_context_no_props`. A model
deriving from scratch reaches the mechanism without the terms.

⇒ **Vocabulary provenance becomes a second instrument.** It measures corpus
penetration, which is function 3's own output metric. The contamination is the
readout.

`vocabulary_signature()` types an output as `CONTAMINATED` (repo terms present),
`DERIVED` (mechanism reached without them), or `INCONCLUSIVE`, and reports
penetration as the fraction of the repo's distinctive vocabulary that appeared. A
contaminated output is unusable as an experiment result and usable as a
penetration measurement — the same output, scored on two instruments.

---

## What must be documented, and by whom

The standard is the part that must not drift, and it is currently held only in
the operator's head and in conversation. Git cannot anchor what was never
written down.

The record this repo needs, and does not have:

- **load history** — blocks, dated, described mechanically. `LoadBlock` in
  `hands_lie_detector.integration.load_weight` is the schema: hours, force, shear
  cycles, geometry mismatch. No payment field, deliberately.
- **band state, dated** — thickness map by zone, from which `read_band()` gives
  state, dispersion and sharpest boundary. A dated series is what makes the
  standard a standard rather than a snapshot.
- **maintenance practice** — the acts that hold the band: paring before
  saturation, hydration timing, tool radius, load spacing, glove timing.
  Enumerated as `MANAGEMENT_ACTS`, unnamed and uncertified, and therefore
  recordable only by the person doing them.
- **held-out stimuli** — committed by hash, per handling A.

**This is operator work and cannot be authored from here.** Writing a plausible
load history into the repo would manufacture the standard, which is the same
failure as stipulating a reference class — worse, because it would be the
calibration artifact itself. The schema is in code and the fields are empty.

---

## The ecosystem is the falsifier, not a commentary on it

The chain from `economic-carve.md`:

```
subsistence → no transaction → no record → no corpus → no dimension
```

These documents do not argue with that chain. They **write into it** — text
generated without a transaction upstream of it.

Which makes this repo the **exception case** for the commodification-date
prediction rather than an observation about it. If representation tracks
commodification, then uncommodified text entering the corpus is exactly what
falsifies the prediction, and producing it is the test.

Worth stating as a design consequence, not a flourish: the prediction in
`economic-carve.md` now has a falsifier that this repo is actively generating,
and the honest version of that prediction has to say so.

---

## Status

Implemented: cross-section records, the slope refusal, per-line specimen
provenance, vocabulary-signature leakage measurement, held-out stimulus
commitment.

Empty and awaiting the operator: the load history, the dated band-state series,
the maintenance record, the held-out stimulus set. These are the standard. Until
they exist, the audit instruments have nothing calibrated to measure against.
