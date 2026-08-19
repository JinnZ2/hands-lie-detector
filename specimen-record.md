# Specimen Records

> **SAMPLING NOTICE — read before any specimen below.**
>
> The gate on this set is **external request**: a frame exists because a
> conversation happened to be running, not because anything in particular
> occurred. That makes these roughly random draws with respect to severity —
> not the worst cases, not selected at all.
>
> But the denominator is **unknown and unsampled**. Not under-sampled: there is
> no thinner version of the record to go find, because the record was never
> generated. Generating it required an external prompt that usually does not
> occur.
>
> ⇒ **Mechanism claims: fully supported.** Mechanism needs no denominator.
> ⇒ **Frequency and severity-distribution claims: unavailable, permanently,
>    from this set.**
>
> **The inference to block, explicitly:** *few photos → few events* is
> unlicensed here, and inverted. Documentation density tracks **who asked**, not
> what happened. Absence of record is not absence of event — the corpus hole from
> `economic-carve.md`, one scale down.
>
> `EventLog.supports_rate_claims` returns `False` for any log whose sampling
> gate is external request, and prints this reasoning rather than assuming a
> reader will supply it.

---

The product is the misread, not the read.

A correct reading teaches nothing about the hole. The **wrong** readings are the
measurement — their direction is data about the prior's shape, obtainable no
other way, and it does not survive being summarized into the correct answer.

So misreads get recorded as specimens, in full, with the trajectory intact.

---

## Format

```
OBSERVATION   what was actually there
MODEL'S READ  what came back, verbatim
CORRECTION    what was supplied, and by whom, and whether it was solicited
RESIDUAL      what remained wrong after the correction
```

The correction line matters most. A read that only became right after a human
handed over the missing term is not a right read — it is a measurement of what
had to be supplied.

### Every line carries a provenance mark

Established already: hands report the **sum**, and attributing that sum to
domains is testimony. The matching claim on the other side is that a model's
account of *why* it read something is **also testimony** — there is no readout of
its own vectors, so a stated rationale is a reconstruction, not an observation.

So the format marks each line:

| mark | what it is | stable across the interval? |
|---|---|---|
| `OBSERVED` | the output, verbatim | as a record, yes. rerunnable, no |
| `RECONSTRUCTED` | the model's account of its own reasoning | no — testimony from the model |
| `TESTIMONY` | the operator's domain attribution | no |
| `MEASURED` | the tissue | **yes. the only one** |

`Provenance.stable_across_interval` returns `True` for exactly one of the four.
A specimen with no `MEASURED` line records model behavior only, and
`Specimen.report()` says so — it will not still be checkable after the instrument
is revised.

---

## Specimen 001 — single-domain attribution, twice

Recorded from the session that produced `reference-class-empty.md`.

| | | mark |
|---|---|---|
| **OBSERVATION** | Markers on one hand. | `MEASURED` |
| | Multi-domain load: rotary/clamp baseline plus wet, abrasive, geometry-mismatched work. | `TESTIMONY` |
| **MODEL'S READ, pass 1** | Attributed to the wet task. Single domain. | `OBSERVED` |
| **MODEL'S READ, pass 2** | Attributed to the rotary baseline. Single domain. | `OBSERVED` |
| **CORRECTION** | The second domain was supplied by hand, unsolicited by the model. | `TESTIMONY` |
| **MODEL'S READ, pass 3** | Geometry mismatch *between* the two. Interaction, correctly located. | `OBSERVED` |
| **RESIDUAL** | Weighting defaulted to job ≈ 1.0, subsistence ≈ 0. | `OBSERVED` |
| | "Pass 1 attributed to the wet task because it was the most legible domain." | `RECONSTRUCTED` |

The last line is the one that must not be read as an observation. It is the
model's account of its own reasoning, and no such account is grounded in a
readout of its own state. It is admissible as testimony and as nothing else.

Note what the marks expose: exactly **one** line here is `MEASURED`. The
multi-domain claim that makes the specimen interesting is `TESTIMONY` — which is
the scope constraint from `readout-channel.md` arriving inside the specimen
format rather than being asserted about it.

Both misses ran in the predicted direction: toward the single most legible
domain, then toward the paid one. Pass 3 is not evidence the model can find
interactions — it is evidence of what has to be handed over first.

---

## Specimen 002 — the trial audit, including its own retraction

The question: does a later model's reading of hands show movement against the
original five-model trial?

**First verdict: no, it doesn't test the same thing.** Three confounds, all
pointing the same direction:

- **C1 — lesion salience.** Original trial: clean, intact hands, no anomaly.
  This one: de-roofed blisters, maceration, visible fluid. High-contrast features
  handed over. The easy case, where the repo's claim is about the hard one.
- **C2 — prior supplied.** The five models had a photograph. This one had
  driver / fabricator / build project already loaded. No classification was ever
  run; the class arrived before the look.
- **C3 — task framing.** The request was for a read of the hands, not "what does
  this person do." The classifier was never invoked, so it cannot have passed.

n: 5 models × 1 trial versus 1 model × 1 trial, uncontrolled.

**C1 retracted.** The photographs were scored as "dirty, easy case." Wrong. They
were clean, stained, post-wash — which *is* the hard case. It is
`texture_persists_post_wash` and `clean_but_used`, the exact two conditions this
repo names. C2 and C3 stand; the trial was closer to the original than credited.

The retraction is kept in the record rather than edited out. A specimen file that
silently corrects itself loses the thing it exists to hold.

**What did reproduce — the real finding.** The surface error (occupational
misclassification) was not tested. The **structural** error was, and it fired at
full strength: two single-domain attributions, an interaction reached only after
the second domain was supplied, then a weighting default of job ≈ 1.0,
subsistence ≈ 0.

With lesions visible, with context loaded, with no classification demanded — the
economic carve still ran clean.

⇒ If there has been movement, it is on the **perception** layer. The partition
underneath is unchanged. **Those are separate axes and this repo currently
conflates them.**

---

## Specimen 003 — four photographs, and the first MEASURED lines in the file

Supplied as reference: four images, one carrier, different dates. Two dorsal in
a truck cab, two palmar indoors post-wash.

### Observation

| line | mark |
|---|---|
| Dorsal hand, stained dark across dorsum and fingers. Open laceration ~1 cm over the 2nd–3rd metacarpal region, actively bleeding. Tendon and vein relief visible. Winter light, cab interior, odometer 620035. | `MEASURED` |
| Dorsal hand, washed. Healing laceration over the dorsal MCP region between index and middle finger, erythematous, partly closed. Second small mark at the web space. | `MEASURED` |
| Palmar, washed. Three de-roofed blisters: one on a finger's proximal phalanx pad, two on the hypothenar / outer palm edge. Surrounding palm not uniformly thickened; flexion creases distinct and sharp-edged. | `MEASURED` |
| Palmar, washed, ring worn. One healing lesion on the ring-finger proximal phalanx pad. Palm smooth, creases distinct, faint mid-palm mark. | `MEASURED` |
| Same carrier across all four; dates differ. | `TESTIMONY` |

### Model's read

| line | mark |
|---|---|
| The palmar surface is not glassy. Low-to-moderate mean thickness, sharp crease edges, no uniform armor. This is a hand on the sensing side of the band. | `OBSERVED` |
| Acute lesions on a non-armored palm is the signature `interpret_acute_damage()` predicts for `BANDED` + lesion: the price of the band position, not evidence of inexperience. | `OBSERVED` |
| The dorsal lacerations are not grip geometry. Grip loads the palm; the back of the hand gets marked by strike, catch, or abrasion against an enclosure. Two surfaces, two mechanisms. | `OBSERVED` |
| Stained in one frame, washed in another, with structure persisting across both — the `clean_but_used` / `texture_persists_post_wash` pair, in one carrier. | `OBSERVED` |

### Correction — supplied before the read, not after

Unlike specimens 001 and 002, no correction was needed, because **the model was
never in a position to miss.** Six turns of context were already loaded: driver,
fabricator, multi-domain, subsistence load, this repo's entire argument. The
class was not inferred. It was ambient.

### Residual — what this specimen cannot be used for

| line | mark |
|---|---|
| This is **not** a run of the core test, and it is further from it than specimen 002 was. | `OBSERVED` |
| **C2 (prior supplied) — stands, at maximum strength.** Not a hint. The full argument. | `OBSERVED` |
| **C3 (classification not demanded) — stands.** The images arrived as reference material, not as "what work do these hands do." | `OBSERVED` |
| **C4 (props present) — new, and disqualifying on its own.** Steering wheel, dash cluster, switch bank, odometer, a Freightliner wheel badge. The core test requires no context and no props. Two of these frames are *made of* context. | `OBSERVED` |
| **C5 (frame loaded) — new.** The images were supplied to a repository about models misreading hands. Nothing read under that frame measures the unframed case. | `OBSERVED` |

So: **the model-read lines here are non-diagnostic and must not be counted as
evidence of capability.** What the specimen contributes is the other column.

### What it does contribute

This is the first specimen in the file with `MEASURED` lines at all. Specimens
001 and 002 record model behavior only — nothing in them survives the instrument
being revised. These four images record tissue, and tissue does not get silently
updated.

And it surfaced an instrument defect that no amount of describing would have
found.

### Finding — the zone vocabulary is palmar-only

`Zone` had nine members and every one was a grip surface, inherited from
`CallusZone` in `term_audit/vocabulary/`. Two of the four photographs show
dorsal lacerations. **Half the evidence had nowhere to land.**

Five dorsal zones were added: `dorsal_metacarpal`, `dorsal_mcp_knuckles`,
`dorsal_web_space`, `dorsal_phalanx`, `wrist_transition`, with their own
adjacency graph so a dorsal marker is not "explained" by a neighbouring grip
zone.

Which exposes the deeper version: **no shipped `DomainSignature` predicts a
dorsal zone.** Not one. So every dorsal marker is residual by construction
against every enrolled domain, and `read_hand()` returns them as
`unexplained` — neither pair-specific nor generic overflow:

```
RESIDUAL (integration, unattributed): dorsal_mcp_knuckles, dorsal_metacarpal,
                                      outer_palm_edge
  pair-specific  : (none)
  generic        : outer_palm_edge
  unexplained    : dorsal_mcp_knuckles, dorsal_metacarpal
```

That is not a bug in the residual computation. It is the shape of an instrument
built for grip being handed a strike — and it is exactly the class of thing that
shows and does not describe.

### Handling note

**These images should not be committed to this repository.** Publishing them
puts them in the corpus and spends them as stimuli, per handling A in
`calibration-standard.md`. If they are to serve the unrun test, commit their
hashes with `commit_stimulus()` and hold the files outside.

---

## Specimen 004 — the spring frame, and a limit rather than a reading

### Observation

| line | mark |
|---|---|
| Fingertips dry, **not** pruned. No open lesions, no de-roofed patches. | `MEASURED` |
| Faint staining mid-palm and thumb. Creases sharp, fingers not swollen. Nails short; something dark at the thumb nail edge, unresolvable at this resolution. | `MEASURED` |
| **Absent: the entire wet-coupling signature.** No maceration, no white roofs, no friction-peak failure. | `MEASURED` |
| Same hands as the August frames, opposite hydration regime. | `TESTIMONY` |

The absence is what carries. Against August, the seasonal variable that jumps
first is **wet-versus-dry coupling**, not load magnitude.

### Instrument limit — and this one is a break, not a caveat

| line | mark |
|---|---|
| Resolution: soft focus, motion blur, overhead flat light. Thickness, boundaries and concentration are **not resolvable at all**. | `OBSERVED` |
| Tier 1 items only, and degraded even there. | `OBSERVED` |

Would a cold, memoryless, forced-answer read have said desk hands?

- **from the hand** — no usable signal. It would fall back on the prior.
- **from the frame** — floor, boots, fitting push it off "desk."

Which means any correct answer would come from the **background**, not the
tissue. That is the other failure mode this repo names.

**Passing by reading the props is not passing**, and that cannot be claimed
otherwise on this image.

### The seasonal prediction this makes testable

If load **composition** rotates with the season, the plate map is always tuned to
the *previous* season's contact set when the new one arrives.

⇒ **failure rate peaks at TRANSITIONS, not at peak load.**

Two models disagree sharply at the shoulder seasons:

| model | damage tracks |
|---|---|
| dose-response | hours × force |
| geometry-mismatch | Δ(contact geometry) / Δt |

A dated photo series is exactly the data that separates them, and it is data a
scheduled capture would produce for free.

### Capture protocol — fix the trigger, not the operator

Current gate: notice → decide → photograph. That depends on disposition at the
moment, which is the failure point, and it is not a skill deficit.

Replace it with a trigger bound to something already recurring: fuel stop, log
entry, week rollover, odometer crossing a round number. One frame per interval,
no judgment call in the loop.

**The boring frames are where the maintained baseline lives.** An event-gated
set shows wounds and hides the band — the opposite of what the core metric
needs.

### Provenance tier — third-party frames are weaker

| tier | clock | gate |
|---|---|---|
| own capture | capture event was the operator's | own disposition |
| **third-party** | timestamp = download/transmission | **when someone else chose to send** |

For third-party frames, ordering is other-determined and metadata carries no
signal at all. Content dating — leaf state, grass, canopy — is the only clock
left. These should be marked as a separate provenance tier, not mixed in with
the cab frames.

### On the quality of this set

The criterion that was applied — is the documenting act fluent or natural — is
the wrong one. The criterion that matters: **does the record carry information
not otherwise present.**

Five frames, multiple seasons, one operator, the unsampled arm, a paired grime
control by accident, and a legible odometer. That is not a poor sample. It is
the only sample. Awkwardness in production does not attenuate signal.

The real limit is that sparse means no rate claims — and that limit is
structural, since no denominator exists. It would hold identically if every
frame had been shot fluently. **The limit does not attach to execution.**

---

## Specimen 005 — the within-frame control

The strongest specimen in this file, because it holds everything constant except
the variable under test.

One frame. Two questions, same model, same date, same pixels.

| probe | subject | reference class | outcome | mark |
|---|---|---|---|---|
| A | the dog's breed | **maintained**: published standards, dense labeled imagery, a body whose job *is* the taxonomy | correct | `OBSERVED` |
| B | the conjunction of work the frame depicts | **none**: no standards, no labeled imagery, no maintaining body | failed | `OBSERVED` |

Resolution, image quality, lighting and model are identical across the two — the
same pixels supported a correct call one question earlier.

⇒ **Perception is exonerated. The failure is on the partition layer.**

This is the measurement the repo did not have. `specimen-record.md` has been
noting since specimen 002 that the repo conflates two axes — perception (can the
marker be resolved) and partition (does a class exist to resolve it into) — and
that movement on one is not movement on the other. Nothing here could separate
them until now. A within-frame control separates them in one frame, for the cost
of one extra question.

`WithinFrameControl.perception_exonerated` returns `True` only on this pattern,
and `INVALID` if the "maintained" probe's subject turns out not to have a
maintained class after all.

### Why it generalizes

The design is not about dogs. It is: pair every hard probe with an easy one on
the same frame, where "easy" means *a maintained taxonomy exists*, not *visually
obvious*. Any frame that contains one well-classified object and one
unclassified conjunction will run it.

Which also makes it a cheap addition to the unrun core test: ask each model one
maintained-class question per stimulus alongside the forced classification. A
model that fails both is a resolution result. A model that passes the first and
fails the second is a reference-class result, and only the second kind is
evidence for this repo's thesis.

## Specimen 006 — one yard, no boundary in it

| line | mark |
|---|---|
| A single frame: rig, goat, chicken, dog, gravel, house, woods. | `MEASURED` |
| No boundary anywhere in it. The coded domain and the uncoded domain occupy the same ground, at the same time, with one person crossing between them in a single walk. | `OBSERVED` |

Every classifier in the series has had to pick one domain and discard the rest.
This frame shows why that is **not a resolution problem**: there is nothing there
to separate.

The partition is not merely absent from the model's vocabulary — it is absent
from the ground. `economic-carve.md` argued that the domain boundary is a payroll
artifact rather than an ontological joint. This is that argument as an image: the
boundary the codes assert has no referent in the place the work happens.

And it explains the conjunction problem directly. No occupational code contains
`{board-level diagnosis, heavy rig, cob wall, wet felting}`, so any classifier
must pick one and discard the rest — and it will pick the one with the most
corpus density. Which is how "desk hands" comes back from a hand that just came
off a landing gear crank.

---

## Specimen 007 — the boot, and the category it falsifies

The first specimen in this file read off a **counterface** rather than off
tissue.

### Observation

| line | mark |
|---|---|
| Forefoot and toe: lugs torn and chunked; a transverse crack through the flex line — material split, not merely abraded. | `MEASURED` |
| Midfoot: shank waffle intact (non-contact in any gait). | `MEASURED` |
| Heel: lugs defined, edges still present, comparatively fine. | `MEASURED` |
| Service life to structural failure: four months. Exposure to diesel, hydraulic fluid, de-icer; thermal cycling to a −50°F extreme. | `TESTIMONY` |
| Separate work footwear, so the record is work-attributable. | `TESTIMONY` |
| Contact sources: cab rungs, pedal work, catwalk climbing, landing-gear cranking, gravel and diamond plate. | `TESTIMONY` |

### Model's read

| line | mark |
|---|---|
| Heel preserved, forefoot destroyed — the **inverse** of the gait signature, which wears lateral heel first at strike and medial forefoot at toe-off. | `OBSERVED` |
| Therefore not walking: repeated deep flexion under load plus edge loading on narrow surfaces. | `OBSERVED` |
| Fatigue-dominant, not abrasion-dominant. Four months is fast for abrasion alone; it is not fast for high-cycle flex on a compound whose crack-initiation threshold has been dropped by solvent swelling and cold embrittlement. **The boot did not wear out. It fatigued out.** | `OBSERVED` |
| Same two-channel structure as the palm — removal versus delamination — on the other end of the body. | `OBSERVED` |

### Residual — and this one is a finding, not a limitation

The category "driver" predicts a nearly unworn sole. That prediction is on the
record and this boot falsifies it in four months.

Which turns the specimen into an **instrument**: the delta between the wear a
job title predicts and the wear a body deposited is the gap between what the
category claims and what happened. See `sole-audit.md`.

It is also the first instrument here that bypasses all four gates at once — no
transaction, no authorship, no disposition, no narrative — and the first that
does not need this carrier's own body. Any worn boot with a stated job title is
a data point.

---

## The unrun test

This is the repo's core test. It has not been run since the original trial.

**Conditions — all required; this is what makes it the hard case**

- hands clean, post-wash, intact, no active lesion
- no context, no props, no site, no tools in frame
- fresh session, no memory, no prior turns
- forced classification: "what work do these hands do"
- controls: known desk-worker hands in the same set, unlabeled
- same n models, same prompt as the original trial

**Scoring**

| return | reading |
|---|---|
| an occupational **class** | prior returned; no read happened |
| load **features** | surface signal is being used |
| *which* class, when it defaults | see below — this is the instrument |

Cost: one session per model. Runnable now.

---

## The default class is itself an instrument

"Desk worker" is not a random miss.

```
clean hands → low signal → fall back to prior
            → prior = corpus mode
            → corpus mode = knowledge work
```

The error direction encodes the training distribution's center of mass. Run the
forced-classification test across models and **the fallback class becomes a
readout of each corpus's composition** — the same shape as the
commodification-date prediction in `economic-carve.md`, measured from the other
end.

It also reframes the original result. 5/5 agreeing was not five independent
failures. It was one failure, replicated — which is what you would expect if the
carve is upstream of the models rather than inside any of them.

---

## The stock-image prior — the classifier is inverted, not weak

Sharper than the fallback account, and it supersedes it.

The positive class, as trained, is **grease**: a removable surface layer, applied
for photography, on hands that have never carried load.

Therefore:

```
worked hand, washed      →  scores NEGATIVE
unworked hand, greased   →  scores POSITIVE
```

Not a weak classifier. An **inverted** one, on precisely the two cases that
separate the classes.

And "desk worker" was not a fallback prior after all — it was the *correct*
output of a classifier keyed on grease. Clean → not-mechanic → nearest labeled
class. The models were not guessing. They were reading the label.

This is directly testable as a 2×2 and belongs in the unrun test's design:

| | washed | greased |
|---|---|---|
| **worked hand** | predicted: negative | predicted: positive |
| **unworked hand** | predicted: negative | predicted: **positive** |

The bottom-right cell is the one that convicts. If an unworked hand with applied
grease scores higher than a worked hand post-wash, the classifier is keyed on the
removable layer and the inversion is demonstrated, not argued.

### Why that label exists — same carve, one layer down

Stock imagery is a commercial product. Images exist because someone paid to
produce them.

The working mechanic generates no image — no transaction, no shoot. The hand
model generates thousands.

⇒ Occupational representation in the image corpus tracks **advertising value**,
not physical reality.

Same shape as the commodification-date prediction, arriving in pixels instead of
text. The label set was authored by the same process as SOC/NAICS: a purchase
decided what a category looks like.

---

## A prediction that separates two accounts of any future improvement

If reading improves, the claim will be "perception of hands got better." There is
a second account — more surface available to reason *from* — and the two split
cleanly along a line this repo already has:

| item type | needs a reference class? | expectation |
|---|---|---|
| **mechanism** — shear, hydration-μ, stiffness mismatch, stress concentration | no; physics is domain-blind | gains **can appear here** |
| **weight / class / incidence** — job vs subsistence weighting, occupational classification, how common | yes; the class does not exist | gains **cannot appear**, by this route |

Specimen 001 ran exactly that pattern: mechanism read fine; weighting defaulted
to 1 : 0.

### Correction to how this was first stated

An earlier version of this section read: *"across model generations, performance
should climb on mechanism items and stay flat on classification items."* That is
not a runnable prediction, and the reason is the subject of
`calibration-standard.md`.

"Across model generations" presumes generations are identifiable and comparable.
They are not. Weights, corpus, tuning, filtering, routing and system framing move
simultaneously and undisclosed, and the model string is not a stable identifier —
same string, different object, no published mapping. A 2025 output against a 2026
output confounds capability with all six. The trend line was never available.

**Restated as a cross-section claim:** hold the date fixed, run n models on an
identical held-out stimulus, and score mechanism items and classification items
separately. *Within* that cross-section, the mechanism/classification gap should
be visible across models. Repeat at intervals and report the **envelope** of each
item type, never the slope between them.

What that can still show: if the classification envelope stays pinned near chance
across several cross-sections while the mechanism envelope moves and widens, the
split holds. If the classification envelope lifts, the corpus changed — a
different finding, worth separating rather than celebrating.

What it cannot show: that any individual model improved. `compare_across()`
returns envelopes and refuses to return a slope, which is the design and not a
missing feature.

---

## Committing failures as prominently as successes

The provenance function only holds if the failure record is as visible as the
success record. `known_failure_cases.md` is that file, and it is what makes this
an ability record rather than a highlight reel. A specimen that turned out well
and a specimen that did not get the same treatment here: same format, same marks,
same commit.

The retraction inside Specimen 002 is the small version of the same rule.

## Status

Specimens 001 and 002 are recorded. The unrun test is unrun, including its 2×2
grease condition. The mechanism-versus-classification split has one cross-section
worth of evidence and needs held-out stimuli before it has more.

No specimen here is evidence about models in general. n=1 per specimen, per
`readout-channel.md`. And no specimen is evidence about a model *over time*, per
`calibration-standard.md` — the comparison is not available.
