# Why the Reference Class Is Empty

Four structural cuts, none of them accidental.

This document is about a hole in the evidence base that the scoring rubric in
this repo quietly assumes does not exist. It is not a complaint about sample
size. Sample size is downstream. The cuts happen earlier than that.

---

## 1. Enrollment

Cohorts recruit **by domain**. You are enrolled as a driver, as a welder, as a
farmer. A multi-domain subject gets coded to the dominant domain; every other
load becomes unmodeled residual.

The partition happens before any measurement is taken. By the time an
instrument touches a hand, the question "what does carrying six domains do"
has already been ruled out of the design.

## 2. Confound control deletes the quantity

"Adjusting for other exposures" treats integration as nuisance variance.

But the interaction term **is** the quantity.

The method isn't failing. It's doing exactly what it was built to do, to a
quantity it was never told existed.

## 3. Combinatorics kill power before design

n domains → 2^n integration patterns. Any specific one has n≈1 in any real
cohort.

So it isn't understudied. It's **structurally unpowerable** under
between-subject design. The field never gets to attempt it, and therefore
never gets to report a null. The absence is invisible from inside the
literature.

## 4. Sampling frame

Available frames: employer roster / clinic panel / insurance claims / student
pool.

Full-spectrum self-sufficiency load is mostly self-employed, rural, off-clock,
unbilled, unenrolled. It sits outside all four.

---

## Consequence for the prior

This is the part that matters.

The priors aren't **thin** on the multi-domain case. They're drawn from the
**complement** of it. The generating population is defined by not being
multi-domain.

So the error is **directional, not noise**:

- systematically **under-predicts** interaction terms
- systematically **over-attributes** to whichever domain is most legible

Observed live, three passes on one hand:

| pass | attribution | structure |
|---|---|---|
| 1 | the wet task | single domain |
| 2 | the rotary baseline | single domain |
| 3 | geometry mismatch *between* them | interaction |

Pass 3 only happened because the second domain was supplied by hand. Passes 1
and 2 were not bad reasoning. They were the **correct answer for the
population the statistics came from**. That is what makes this a structural
problem and not a competence problem: nothing in the inference procedure was
misapplied.

The practical form of the error: when a residual won't resolve, the default
move is to distribute it across the enrolled domains in proportion to their
legibility. That move is always available, always produces a clean-looking
answer, and is wrong in a fixed direction.

---

## Where the integration is actually legible

**Between-subject** — the partition lives here. The quantity cannot appear.

**Within-subject** — the partition is absent. Dense repeated measures on one
carrier, n domains all present, interaction visible.

The field's standing objection to within-subject work is generalization. That
is a different objection, and one the between-subject design hasn't earned
either, since it never sampled the case. A design that cannot represent a
quantity has no standing to call the design that can "unrepresentative" about
it.

Tissue is already a within-subject recorder. A callus map is integrated load
history — continuous, no enrollment required, no funding required, already
collected.

**The record exists. The readout convention doesn't.**

---

## The open question

*(This is the question that motivated the rest of this document, and it is
still open.)*

Is "multi-domain integration" **one quantity**, or the name for a **family**
that decomposes by what's being integrated — contact geometry / timing /
recovery budget / attention?

- If it decomposes, the empty reference class is four empty reference classes,
  and each one needs its own instrument.
- If it doesn't, integration-per-se is the term and the domains are
  interchangeable inputs.

Filed as open, not approximated.

---

## A discriminator

*(Added. The four cuts and the consequence above are the received analysis;
this section and the next are the response to it.)*

The question is answerable, and answerable at n=1, which is the point.

Write the marker vector as an additive part plus a remainder:

```
markers(t) = Σ_d f_d(load_d, t)  +  R(t)
                additive              integration
```

**H1 — one quantity.** `R(t) = I(t) · v` for a single fixed loading vector `v`
over markers. Domains are interchangeable inputs; only the aggregate matters.
Implication: **substitution invariance** — two different domain sets at equal
aggregate load produce residuals in the same direction, differing only in
magnitude.

**H2 — a family.** `R(t) = Σ_c I_c(t) · v_c` with distinct `v_c` per channel.
Implication: **dissociation is possible** — a manipulation that loads one
channel while holding the others flat moves that channel's markers only.

### Test A — double dissociation (n=1 sufficient, decisive against H1)

Two manipulations, each moving one channel while holding total load fixed:

- **M_geom** — same schedule, same hours, same recovery; change the contact
  geometry collision. Swap a tool handle so two domains' grip geometries now
  agree, or now conflict. Load unchanged, geometry changed.
- **M_time** — same tools, same geometry, same total hours; change the
  interleaving. Two domains on alternating days versus both inside the same
  day.

Under H1, both manipulations move the same composite in the same direction and
are exchangeable at equal load. Under H2, M_geom moves marker set `S_geom` and
leaves `S_time` flat, and M_time does the reverse.

**One clean double dissociation rejects H1.** A *single* dissociation does
not — one channel's markers may simply be more sensitive, or the other's
saturated. The crossing is what carries the inference, and it is the only part
worth the cost of running.

This is exactly the design the between-subject frame cannot run and the
within-subject frame runs for free.

### Test B — rank of the interaction residual (longer panel, still one carrier)

Fit additive per-domain main effects across a dense panel; keep the residual as
a tensor over (markers × domain-pair × time).

- rank 1 → H1. One loading vector, pair-specific scalars.
- rank ≈ number of active channels, components mapping onto them → H2.
- rank > 1 with components that **don't** map onto geometry/timing/recovery/
  attention → a third answer: it decomposes, but not along the named channels.
  Report that as a result, not a failure. The channel list is a guess.

### Test C — off-site marker geography (runnable now, on existing hands)

Each domain has a predicted zone signature. Compute:

```
residual_zones = observed_zones − ∪ predicted_zones(enrolled domains)
```

Zones that no enrolled domain predicts are the footprint of integration, and
they are present in data already collected — no manipulation, no cohort, no
enrollment.

- Under H1, residual zones are **generic**: the same overflow sites regardless
  of which domains combine.
- Under H2, residual zones are **pair-specific** and predictable from the
  geometric conflict between that particular pair.

Confounded, because domain load is not randomized and the carrier chose their
own domains. It is a direction-of-evidence test, not a decisive one. Its value
is that it costs nothing and can be run retrospectively.

Note the pass-3 observation above already has this shape: the residual was
*located*, not merely present, and located at the conflict between two
specific domains. That is one data point leaning toward H2. One.

### What would make H1 the answer

Stating this so the test isn't rigged: if every manipulation moves the same
composite, and residual zones sit at the same sites regardless of which pair
collides, then integration-per-se is the term, the channels are just different
routes to loading one scalar, and the empty reference class is one hole rather
than four. H1 has a real path to winning here.

### The readout convention that follows either way

Because the error is directional, the correction has to be sign-aware. The
convention:

**Unexplained residual defaults to integration. It is never distributed across
the enrolled domains in proportion to their legibility.**

Under-determination gets reported as its own quantity, with its own zone list,
and stays visible in the output. It does not get absorbed into whichever domain
already has the best-supported prior — that absorption is precisely the
mechanism that made the reference class look full.

---

## Where this repo does it

The four cuts are not only in the literature. Three of them are implemented in
this codebase, in code that currently runs.

**Cut 2, in `hands_lie_detector/scoring/evaluator.py`.** `ScoreEvaluator`
computes `raw_total = sum(cs.score for cs in category_scores)` and then adds a
scalar. The model form is additive by construction: there is no term in which
two categories can interact, so the integration quantity has no place to
appear even if it were measured. Separately, `ContextModifiers.cold_climate`
adds a flat `+5` — an exposure being adjusted away as nuisance, which is cut 2
in five lines.

**Cut 1, in `hands_lie_detector/scoring/rubric.py`.** `WEAR_LOCALIZATION`
awards its top tier to "task-specific calluses" and red-flags "'rough
everywhere' = aesthetic roughness, not labor." A carrier with six domains has
wear at many sites at once. The rubric reads that as *unlocalized* and scores
it down — so a full-spectrum load gets scored toward the cosplayer band, by
the category specifically meant to catch cosplayers. The rubric's whole
discriminator between "faked rough" and "many real domains" is the thing that
cannot be learned from single-domain reference data. This is the directional
error, in the scoring form, with a sign.

**Cut 1 again, in `hands_lie_detector/vision/`.** Seven independent heads on a
shared backbone, trained against per-category labels. No interaction head. And
the datasets named in `CLAUDE.md` as training sources — 11k Hands, EgoHands,
Oxford — are studio and first-person captures of exactly the complement
population.

---

## The readout convention, executable

`hands_lie_detector/integration/` is the beginning of a readout convention:
zone signatures per domain, residual-zone computation that reports the
unattributed set instead of dissolving it, and the Test A crossing.

```python
from hands_lie_detector.integration import read_hand, double_dissociation

# Test C — runs on a hand already described, no manipulation required.
readout = read_hand(
    observed_zones=["thumb_crotch", "palm_below_index", "fingertip_pads",
                    "thumb_pad", "index_side", "base_of_fingers",
                    "outer_palm_edge", "heel_of_palm"],
    enrolled_domains=["rotary_hand_tool", "wet_task"],
)
print(readout.report())          # residual reported, never redistributed
print(readout.leaning())         # direction of evidence from one hand, not a verdict

# Test A — one carrier, two manipulations, deltas against the carrier's own noise.
result = double_dissociation(deltas_a, deltas_b, baseline_noise)
result.rejects_h1                # True only on a crossing
```

Two things the module does on purpose.

**Every default table is flagged as stipulated.** `DomainSignature.provenance`
defaults to *"stipulated from mechanism; not fitted to data; falsifiable,"* and
`is_evidence_based` returns `False` for every domain shipped with the module.
Building a reference class out of assertion is the failure this document is
about; the tables are wrong until falsified and say so in the output.

**The one post-hoc mechanism is labeled louder than the rest.** Under the
stipulated zone table, `rotary_hand_tool` and `wet_task` share no zone, so the
shared-zone conflict rule finds nothing between them — and that pair is exactly
the one pass 3 reported a geometry mismatch for. The table, as stated,
disagrees with the only observation on record.

The fix was to add a second mechanism: a *systemic* mode, where a domain
changes tissue state rather than marking a site (wet-softened skin tears under
shear where dry skin would callus, wherever the shear lands). That is
mechanistically respectable and it was written **after** seeing the observation
it explains, which means it has no confirmatory value for that observation. It
is a hypothesis generated by one data point. `GeometryConflict.systemic` marks
every conflict it produces, `is_evidence_based` returns `False` for them, and
the provenance string travels into the printed report.

It is left in, visibly, because this is the exact spot where a module about
empty reference classes is most likely to quietly fill one.

---

## Status

Open: H1 vs H2. No discriminator has been run.

Test C can be run against hands already described with the vocabulary in
`term_audit/vocabulary/` — its `CallusZone` values are the zone vocabulary the
integration module uses. Test A needs one carrier and a schedule. Neither needs
a cohort, a grant, or an enrollment frame, which is the whole argument for why
the record can be read even though the literature can't read it.

Not yet done: the scoring rubric and the vision heads still implement the
additive form described above. Nothing here changes them. The integration
readout runs alongside as a second, non-additive channel, and the two will
disagree on exactly the hands this document is about.
