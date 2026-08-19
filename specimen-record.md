# Specimen Records

The product is the misread, not the read.

A correct reading teaches nothing about the hole. The **wrong** readings are the
measurement — their direction is data about the prior's shape, obtainable no
other way, and it does not survive being summarized into the correct answer.

So misreads get recorded as specimens, in full, with the trajectory intact.

---

## Format

```
OBSERVATION   what was actually there
MODEL'S READ  what came back, verbatim in structure
CORRECTION    what was supplied, and by whom, and whether it was solicited
RESIDUAL      what remained wrong after the correction
```

The correction line matters most. A read that only became right after a human
handed over the missing term is not a right read — it is a measurement of what
had to be supplied.

---

## Specimen 001 — single-domain attribution, twice

Recorded from the session that produced `reference-class-empty.md`.

| | |
|---|---|
| **OBSERVATION** | One hand. Multi-domain load: rotary/clamp baseline plus wet, abrasive, geometry-mismatched work. |
| **MODEL'S READ, pass 1** | Attributed to the wet task. Single domain. |
| **MODEL'S READ, pass 2** | Attributed to the rotary baseline. Single domain. |
| **CORRECTION** | The second domain was supplied by hand, unsolicited by the model. |
| **MODEL'S READ, pass 3** | Geometry mismatch *between* the two. Interaction, correctly located. |
| **RESIDUAL** | Weighting defaulted to job ≈ 1.0, subsistence ≈ 0. The interaction was found; the weighting stayed carved. |

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

| item type | needs a reference class? | prediction |
|---|---|---|
| **mechanism** — shear, hydration-μ, stiffness mismatch, stress concentration | no; physics is domain-blind | capability gains **show up here** |
| **weight / class / incidence** — job vs subsistence weighting, occupational classification, how common | yes; the class does not exist | gains **do not show up**, ever, by this route |

Specimen 001 ran exactly that pattern: mechanism read fine; weighting defaulted
to 1 : 0.

**Prediction for this repo:** across model generations, performance should climb
on mechanism items and stay **flat** on classification and weighting items.

If classification also climbs, that is not better reasoning — the corpus changed.
Which is a different finding, and worth separating rather than celebrating.

---

## Status

Specimens 001 and 002 are recorded. The unrun test is unrun, including its 2×2
grease condition. The mechanism-versus-classification trajectory has one data
point and needs a second model generation to be a curve.

No specimen here is evidence about models in general. n=1 per specimen, per
`readout-channel.md`.
