# CLAUDE.md — Hands Lie Detector

## Project Overview

A Python framework for detecting real physical work experience from hand (and foot) markers. The core thesis: vision models confuse "clean hands" with "unused hands," overfitting to dirt/props/context instead of persistent structural adaptation in skin and tissue.

The project includes:
- **Documentation** — Scoring rubrics and failure case analysis (Markdown)
- **Scoring module** — Pure Python rubric logic, evaluation, and context modifiers
- **Vision classifier** — PyTorch multi-head classifier with pretrained backbones
- **Prompt evaluator** — Send hand images to vision LLMs (Claude, GPT-4o) with structured rubric prompts

## Repository Structure

```
hands-lie-detector/
├── hands_lie_detector/           # Python package
│   ├── __init__.py
│   ├── scoring/                  # Pure Python — no dependencies
│   │   ├── rubric.py             # 7-category rubric, bands, category definitions
│   │   └── evaluator.py          # Score aggregation, context modifiers, interpretation
│   ├── vision/                   # Requires: torch, torchvision, Pillow
│   │   ├── classifier.py         # Multi-head CNN (ResNet50/18, EfficientNet-B0)
│   │   ├── dataset.py            # Image dataset loader with labels CSV
│   │   └── train.py              # Training loop with validation and checkpointing
│   ├── prompt/                   # Requires: anthropic or openai
│   │   └── evaluator.py          # Vision LLM scoring with rubric prompt
│   ├── audit/                    # Pure Python — no dependencies
│   │   ├── crosssection.py       # Cross-sections; envelope, never slope
│   │   ├── specimen.py           # Specimen records, per-line provenance
│   │   ├── condition.py          # Condition coordinates; what n=1 supports
│   │   ├── reference_class.py    # Within-frame control; isolates the layer
│   │   ├── attribution.py        # Agent retrofit: 4 tests, no ground truth
│   │   └── leakage.py            # Vocabulary provenance; held-out commitment
│   ├── band/                     # Pure Python — no dependencies
│   │   ├── contrast.py           # Map states; concentration, NOT skill
│   │   └── capture.py            # Five positions; the raking-light protocol
│   └── integration/              # Pure Python — no dependencies
│       ├── domains.py            # Domain zone/load-mode signatures, channels
│       ├── residual.py           # Residual-zone readout (Test C), conflicts
│       ├── dissociation.py       # Double-dissociation discriminator (Test A)
│       ├── load_weight.py        # Mechanical load weighting; no payment term
│       ├── partition.py          # Inverted null: partition must be argued for
│       ├── strip.py              # Render category nouns into mechanical units
│       ├── gated.py              # Layers gate, not add; the reversed arrow
│       ├── event_log.py          # Dorsal marks: events, not load history
│       ├── wear.py               # Tribology taxonomy; counterface required
│       ├── sole.py               # Sole wear audits the job description
│       ├── nail.py               # Third clock: matrix trauma, self-dating
│       ├── knuckle.py            # MCP/PIP/DIP instrument; load modes
│       ├── healing.py            # Residual mark != trauma; 4th sign error
│       └── carve_audit.py        # Retrieval protocol, seam, boundary audit
├── scoring-metrics.md            # Main 100-point hand scoring rubric (v0.1)
├── feet-lie-detector.md          # Parallel scoring system for feet
├── gloves-debunk.md              # Addresses the "I wore gloves" excuse
├── known_failure_cases.md        # Catalogues specific vision model failures
├── reference-class-empty.md      # Why multi-domain load has no reference class
├── economic-carve.md             # Why the partition is payroll, not ontology
├── band-not-scale.md             # Three states; the monotone sign error
├── readout-channel.md            # What the channel bypasses; scope of claims
├── specimen-record.md            # Misreads as the measurement; the unrun test
├── calibration-standard.md       # Hand as standard, model as drifting sample
├── gated-not-summed.md           # Wrong form, not wrong number (x3)
├── contrast-case.md              # What n=1 does; show vs describe; coordinates
├── attribution-retrofit.md       # Agent reassignment; weight vs constraint
├── wear-taxonomy.md              # Tribology spine; deposit vs draw; two clocks
├── sole-audit.md                 # Boot wear vs stated job title; four gates bypassed
├── capture-protocol.md           # Five positions, one lamp; unblocks tier 2
├── knuckle-instrument.md         # MCP as its own readout; unread not misread
├── healing-calibration.md        # Residual mark != trauma; the fourth sign error
├── tests/                        # unittest suite (zero deps)
├── README.md                     # Mission statement and project premise
├── requirements.txt              # All dependencies
├── setup.py                      # Package setup with optional extras
├── LICENSE                       # MIT License
└── CLAUDE.md                     # This file
```

## Installation

```bash
# Core scoring module only (zero dependencies)
pip install -e .

# With vision classifier
pip install -e ".[vision]"

# With prompt-based evaluator
pip install -e ".[prompt]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### Scoring module (no dependencies)
```python
from hands_lie_detector.scoring import ScoringRubric, ScoreEvaluator, ContextModifiers

rubric = ScoringRubric()
evaluator = ScoreEvaluator(rubric)

result = evaluator.evaluate({
    "Texture Persistence": 20,
    "Wear Localization": 16,
    "Micro-Injury History": 12,
    "Tendon & Vein Definition": 11,
    "Nail Evidence": 7,
    "Symmetry of Wear": 8,
    "Climate & PPE Intelligence": 4,
}, modifiers=ContextModifiers(cold_climate=True))

print(result.adjusted_total)  # 83.0
print(result.band_label)      # "Experienced Trade / Field Work"
```

### Prompt evaluator (quick build)
```python
from hands_lie_detector.prompt import PromptEvaluator

evaluator = PromptEvaluator(provider="anthropic")  # uses ANTHROPIC_API_KEY env var
result = evaluator.evaluate("path/to/hand.jpg")
print(result.band_label)
```

### Vision classifier (training)
```python
from hands_lie_detector.vision import HandClassifier, Trainer

model = HandClassifier(backbone="resnet50", freeze_backbone=True)
trainer = Trainer(model, lr=1e-3)
trainer.fit("data/images", "data/labels.csv", epochs=30)
```

## Key Documents

- **scoring-metrics.md** — The primary rubric. Seven scoring categories totaling 100 points. Note: contains a duplicate section starting at ~line 112 (older version without PPE).
- **feet-lie-detector.md** — Analogous rubric for feet. **Carries a design
  correction at the top**: it reads gait, which is dynamic, and a still frame
  discards sequence, variability and perturbation recovery. Needs video, or needs
  to read boot wear as the counterface instead.
- **gloves-debunk.md** — Why gloves don't erase structural adaptation.
- **known_failure_cases.md** — Specific vision model failure modes.
- **attribution-retrofit.md** — Agent reassignment as an **unmeasured
  hypothesis**, not a documented failure class. Carries a provenance header: the
  account was authored by a model describing its own behavior, which is
  `RECONSTRUCTED` under this repo's own scheme — testimony, not observation, and
  not the operator's claim. Specs a severity ladder (fill / fabricate /
  override), four tests, and the weight vs constraint distinction. Three of the
  four tests depend on scoring instruments that do not exist yet; only the
  no-destination test is runnable today.
- **healing-calibration.md** — Residual mark is trauma x (1 - healing quality),
  so a scar count read straight marks a carrier down for healing well. The
  FOURTH sign error, and the same shape as the third: a physiological parameter
  with no load content entering a monotone count uncalibrated. Also names what
  no light angle reaches — fracture callus, capsular thickening, ligament laxity
  — and extends the joint instrument to PIP and DIP.
- **knuckle-instrument.md** — The MCP joint as a separate instrument. Three load
  modes (impact / hyperextension / hyperflexion-under-load), a marker taxonomy,
  and the structural claim that **palmar and dorsal load are not correlated**, so
  neither licenses an inference about the other. Scar LOCATION distinguishes
  posture: MCP scar means the joint was flexed when struck, metacarpal shaft
  means flat or dragged. Not a diagnostic instrument — the pad-versus-bony-node
  differential is a palpation finding and the module declines it.
- **capture-protocol.md** — Five positions and five rules. Position 5 (lateral
  raking) is the one that unblocks tier 2, and nothing else substitutes for it.
  `CaptureSession.problems` reports every rule a session missed rather than
  accepting frames that turn out to be unreadable.
- **sole-audit.md** — The cheapest instrument here and the first that scales past
  one operator. A job description is AUTHORED; sole wear is DEPOSITED, so the
  delta between the wear a title predicts and the wear a body left is the gap
  between the category and the work. Bypasses all four gates at once, needs no
  consent, and already exists on every working person's feet.
- **wear-taxonomy.md** — Wear is classified by mechanism, never application, and
  the taxonomy (adhesive/abrasive/fatigue/corrosive/fretting) is already
  domain-blind. Wear is a SYSTEM property, so the tool carries the conjugate
  record and a hand alone is half a specimen. The band is run-in held. Dorsal is
  a separate instrument on a ~2wk healing clock against palmar's 2-4wk turnover.
  Deposit vs draw vs suppress is the third form error: a sign, not a coefficient.
- **contrast-case.md** — What one characterized point against a dense
  distribution can do (falsify no-axis, give direction, give rough magnitude)
  and cannot (variance, prevalence). Why the measurement must be SHOWN while the
  convention must be WRITTEN, reconciling the two halves. And the scope-line
  correction: arm A's data is not wrong, it is unscoped — every "the literature
  is carved" phrasing in this repo should be read that narrower way.
- **gated-not-summed.md** — The weighting question had the wrong functional form.
  environment → capacity → job is a gate chain, and a weighted sum cannot express
  a gate at any coefficient. Also indexes the same error class three times in this
  repo (additive scoring, monotone scale, summed layers) and gives the diagnostic
  that catches all three: ask what the form cannot represent at any parameter
  setting.
- **calibration-standard.md** — The inversion: the model is a drifting sample and
  the hand is the fixed standard, so the repo measures models, not hands. Why a
  cross-model trend line is structurally unavailable (weights, corpus, tuning,
  filtering, routing and framing all move undisclosed; the model string is not an
  identifier), the cross-section design that survives, the three-function
  conflict between experiment, provenance and development-forward, and the two
  handlings for publication-as-leakage.
- **band-not-scale.md** — Experience is regulation toward a setpoint, not
  accumulation. Three states (soft / banded / glassy); mean thickness cannot
  separate the last two, contrast separates all three. Documents two sign errors
  in the current rubric: saturation reads as expertise, acute damage reads as
  incompetence.
- **readout-channel.md** — Why the tissue route bypasses the gate that deletes
  the quantity, and the three constraints that scope this repo's claims: the sum
  not the decomposition, written conventions not photographs, n=1 fine for
  mechanism and not for anything distributional.
- **specimen-record.md** — Misreads recorded as the measurement, in the format
  OBSERVATION → MODEL'S READ → CORRECTION → RESIDUAL. Contains the unrun core
  test, the stock-image inversion (the classifier is keyed on grease, so a washed
  worked hand scores negative), and the prediction that mechanism items improve
  across model generations while classification items stay flat.
- **economic-carve.md** — Where the domain partition came from: job title →
  SOC/NAICS → risk class → study stratum → literature → prior, with the
  provenance lost by the last hop. Covers the unpaid-domain null, the
  mechanism-vs-incidence retrieval protocol, the null inversion, the boundary-
  alignment discriminator, and the discontinuity test. Escalates
  `reference-class-empty.md` from "directional error" to "absent dimension."
- **reference-class-empty.md** — Why the evidence base for multi-domain load is
  structurally empty (enrollment, confound control, combinatorics, sampling
  frame), why the resulting prior error is directional rather than noisy, and a
  within-subject discriminator for whether integration is one quantity or a
  family. Also audits where this repo implements the same cuts.

## Architecture Notes

### Scoring module (`hands_lie_detector.scoring`)
- Zero external dependencies — works standalone
- `ScoringRubric` defines categories, tiers, and interpretation bands
- `ScoreEvaluator` handles aggregation and context modifiers (climate, age, glove penalty)
- All default values match the v0.1 rubric from `scoring-metrics.md`

### Vision classifier (`hands_lie_detector.vision`)
- Multi-head architecture: shared backbone -> 7 independent scoring heads
- Each head outputs a score in [0, max_points] for its category
- Supports ResNet50, ResNet18, EfficientNet-B0 backbones
- Dataset expects `images/` directory + `labels.csv` with 7 score columns
- Training includes validation split, per-category MAE tracking, and best-model saving

### Audit instruments (`hands_lie_detector.audit`)
- Zero external dependencies
- Companion to `calibration-standard.md`
- `ModelResponse` stores verbatim output only — **no `summary` field**, because a
  summary is an interpretation and the interpretation cannot be re-derived.
  `is_stable_identifier` returns `False` unconditionally
- `CrossSection` rejects mixed dates and mismatched stimuli. `compare_across()`
  returns envelopes and **refuses to return a slope**; that is the design, not a
  gap
- `Provenance` marks each specimen line OBSERVED / RECONSTRUCTED / TESTIMONY /
  MEASURED. Only MEASURED is stable across an interval — a model's account of its
  own reasoning is RECONSTRUCTED, since there is no readout of its own vectors
- `reference_class.WithinFrameControl` is the instrument that separates the two
  axes the repo had been conflating. Two probes on ONE frame, same model, same
  pixels: one subject with a maintained reference class (standards + dense
  labeled imagery + a body whose job is the taxonomy), one without. A pass on the
  first and a failure on the second exonerates perception and locates the failure
  on the partition layer. `perception_exonerated` returns True only on that
  pattern
- `attribution.NoDestinationTest` is the cheapest instrument here and refuses to
  be built on a window that contains a candidate party. In a window with none, an
  invented agent is fabrication rather than bias: binary, not a magnitude, and
  the window is its own control
- `attribution.DoseResponse.classify()` separates a weight (error decays with
  evidence) from a constraint (error is flat). A correction that returns after a
  model update is the constraint case confirmed
- `attribution.InventedAgent.supports_slot_hypothesis` encodes the falsifiable
  prediction: the invented party stays UNNAMED and grammatically necessary. Named
  would mean person-bias; unnamed means a required slot
- `condition.ConditionSpec` requires provision, rate, duration and cadence per
  need; `is_plottable` is False while any is unstated. `ARM_A_UNSTATED` ships
  with all eight coordinates unstated, so the dense baseline fails the same
  check the sparse arm is asked to pass — the missing scope line as a failing
  check rather than an accusation
- `vocabulary_signature()` types an output as CONTAMINATED / DERIVED /
  INCONCLUSIVE and reports corpus penetration. A contaminated output is unusable
  as an experiment result and usable as a penetration measurement
- `commit_stimulus()` records a held-out item by hash; the item is never stored

### Band readout (`hands_lie_detector.band`)
- Zero external dependencies
- Companion to `band-not-scale.md`. The scoring rubric is monotone in thickness
  (seven monotone categories, summed), so it ranks a saturated undifferentiated
  hand highest. This package reads the thickness map for CONTRAST instead
- **RETRACTED: dispersion is not a skill proxy.** It reads the geometric
  CONCENTRATION of the load history. A fixed-geometry specialist scores high and
  a variable-geometry generalist scores low at identical competence — the
  desk-hands error re-entering through the replacement metric
- `UNIFORM_THICK` is AMBIGUOUS by construction: a generalist and a saturated hand
  produce the same map. The separator is a functional sensing test, not an image,
  and `position` stays `UNRESOLVED` until one is supplied
- `BiologicalCalibration` separates the two quantities cleanly: **mean thickness
  is calibrated, band position is not**. The map is normalized before the state
  logic runs, so state and concentration are invariant across baselines while the
  uncalibrated monotone score drops — a THIRD sign error in the rubric,
  compounding with saturation. At a lower baseline a healed lesion at a load
  point is MORE informative, since it formed nearer the sensing threshold
- `nail.NailRecord` is the third clock: palmar integrates over 2-4 weeks, dorsal
  heals in ~2, and the nail plate carries matrix trauma outward for 4-6 months,
  dating its own marks by distance from the fold. It corroborates the palmar map
  rather than restating it. Treat the ORDERING as the finding; the months are
  stipulated
- `ThicknessReading.measurable` is False except in raking light. Backlit and
  overhead-flat field photos cannot resolve thickness at all — see the tier split
  in `band-not-scale.md`
- `monotone_score()` is a stand-in for the rubric's *shape*, not the rubric.
  `BandReadout.monotone_disagrees` flags the sign error and `report()` prints it
- `read_band()` refuses dorsal zones; `DorsalMark` refuses palmar ones. The two
  instruments decline each other's inputs
- `interpret_acute_damage()` inverts the second sign error: a blister on a
  banded hand is the price of the band position, not a demerit
- `Zone` gained thenar, hypothenar and proximal-phalanx-pad after specimen 008,
  for the same reason as the dorsal zones: observations kept landing where the
  vocabulary had no coordinate
- `Zone` gained five DORSAL zones after specimen 003; the vocabulary inherited
  from `term_audit/` was palmar-only and could not record a strike. No shipped
  `DomainSignature` predicts a dorsal zone, so dorsal markers are residual by
  construction and return as `unexplained`
- Reads the integrated map only. It does not attribute zones to domains — see
  the scope constraint in `readout-channel.md`
- Thresholds are stipulated, and say so in their own output

### Integration readout (`hands_lie_detector.integration`)
- Zero external dependencies
- Companion to `reference-class-empty.md`; sits alongside the scoring module
  rather than replacing it
- `ScoreEvaluator` is additive by construction (`sum` of 7 categories), so no
  two categories can interact. This module keeps the non-additive part as its
  own quantity: `read_hand()` returns residual zones — markers no enrolled
  domain predicts — and never redistributes them across enrolled domains
- `geometry_conflicts()` finds zones two domains load in incompatible modes;
  that quantity exists only in the pair, so a domain-partitioned cohort deletes
  it by construction
- `double_dissociation()` implements the n=1 within-subject discriminator;
  movement is judged against the carrier's own baseline variability, which is
  why no cohort is needed
- **All default signature tables are stipulated from mechanism, not fitted to
  data.** `is_evidence_based` returns `False` for every shipped default, and the
  provenance travels into printed reports. Do not let these harden into
  evidence — that is the failure the accompanying document is about
- `load_weight.LoadBlock` has **no payment field**, deliberately: payment does
  not appear in the governing equations, so it cannot appear in the weight.
  `ledger_share()` implements the occupational-hour denomination as the artifact
  under examination — do not use it to weight anything
- `partition.propose_partition()` inverts the null: `LoadHistory` (one body, one
  load history) is the default, and a proposed domain split must clear a
  permutation test against same-grain random splits. Against the current
  8-domain stipulated registry **nothing earns a partition**; that is a statement
  about the registry, not about any split
- `gated.GatedStack` implements the layer chain. Load adds WITHIN a layer
  (`load_weight`'s integral is unchanged and still right) and **gates ACROSS**
  layers. `additive_output()` exists to be wrong next to `output()`, the same way
  `ledger_share()` sits next to `load_share()` and `monotone_score()` next to
  `read_band()` — this repo keeps its errors runnable rather than described
- `gated.arrow_check()` prints the dependency direction in both accountings: the
  record calls the base layers consumption and the top layer production; physics
  has it reversed. A sign error on direction, not a missing coefficient
- `gated.solvency_from_band()` reads capacity solvency off a resolved
  `BandPosition`, not off a raw map state — the map cannot establish solvency
  because it cannot separate generalist from saturated. `UNRESOLVED` returns 0.0:
  unread is not solvent
- `gated.deposit_draw_balance()` sorts domains by which side of the ledger they
  sit on. Deposit writes the map; draw spends or suppresses the sensing capacity
  it produced. Opposite signs, not parallel inputs
- `wear.WearSystem.is_complete_specimen` is False without a counterface. Every
  wear measurement in this repo is half a specimen until tool handles are
  photographed alongside hands
- `event_log.carries_grip_load_history` is False unconditionally — the dorsum is
  not a grip surface, and this is the only part of the original claim that should
  have been stated that strongly. `carries_contact_history` is the narrower
  channel that CAN be True: repeated dorsal CONTACT remodels, and impact,
  friction and pressure all qualify. Corrected twice in the same direction — see
  the table in `knuckle-instrument.md`
- `event_log.dorsal_signature()` separates an EDGE_STRIKE_FIELD (many superficial
  marks at varied sites — confined volume, sudden release, different geometry
  every reach) from REPEATED_CONTACT (few marks, concentrated, with thickening).
  REPEATED_CONTACT is AMBIGUOUS by construction: a striker and a carpet layer
  land in the same class, and the co-occurring scar field is what separates them
- `healing.HealingCalibration.marks_understate_history` is True whenever the
  residual factor is below 1. `implied_events()` returns an unbounded float on
  purpose: the direction is the finding and the coefficient is stipulated, per
  the living-tissue seam
- `healing.BELOW_THE_SURFACE` names what no capture protocol reaches — fracture
  callus, capsular thickening, ligament laxity, tendon adhesion. A scope limit,
  not a resolution problem
- `knuckle.Joint` extends the instrument to PIP and DIP. DIP trauma reaches the
  nail matrix, so `writes_to_nail_clock` prompts the cross-check against
  `nail.py` — a DIP finding and a nail mark at consistent age are two
  instruments on one event
- `audit.SpecimenLine.falsifiable_on_demand` separates a capability claim from a
  description. Both are TESTIMONY; only one can be re-run and fail visibly
- `knuckle.KnuckleReadout.predicts_palmar_load` returns False unconditionally.
  Dorsal and palmar load are not correlated, so neither surface's reading
  licenses an inference about the other
- `event_log.EventLog.supports_rate_claims` is False when the sampling gate is
  external request. Documentation density tracks who asked, not what happened
- `event_log.EventLog.has_baseline_coverage` is False without scheduled frames.
  The narrative gate fires inside every population — heavy documenters record the
  anomaly, not the Tuesday — so steady state is missing from ALL arms and the
  boring frame is the expensive one
- `DomainSignature.is_bundle` marks one word covering several contact
  distributions. "Firewood" is cut/split/stack/base, which is why its deposit
  cannot resolve to a single site
- `strip.unweld()` handles the general form the strip is one case of: a
  cardinality reduction with the reduction step deleted from the output. False
  binary reduces the option set, welded term reduces two variables to one, narrow
  framework reduces the population — one operation at three layers
- `sole.audit()` refuses a job title with no stipulated `CategoryPrediction` on
  file: a category has to make a falsifiable claim about wear before it can be
  audited, and writing that claim down is the first step of the protocol
- `sole.SoleReading.inverted_signature` detects heel-preserved / forefoot-
  destroyed — the inverse of the gait signature, so not walking
- `sole.SoleReading.time_to_failure_supports_distance_claim` is False when the
  failure is fatigue-dominant. A short service life is not a mileage reading;
  treating it as one is the same error shape as reading few photographs as few
  events
- `wear.CONJUGATE_PAIRS` completes the counterface move: hands to tool handles,
  feet to boots. `feet-lie-detector.md` reads gait, which is a sampling mismatch
  — a still keeps the pose and discards sequence, variability and perturbation
  recovery. Boot wear is static, shootable at rest, and self-dating
- `strip.strip()` is the cheapest diagnostic here and should be reached for
  first: render a category noun into force / displacement / cycles / duty cycle.
  Same units on both sides means the category carried no physical information;
  unrenderable means it is a ledger class wearing a physical one. All five
  `DEFAULT_BANDS` labels strip to `LEDGER_CLASS`
- `carve_audit.classify_relation()` fails closed, and types relations five ways.
  **The seam: constitutive parameters for living tissue** (stiffness, fatigue
  limit, adaptation rate, hydration response) return `RELATIONAL_ONLY` — the
  relation transfers, the coefficient does not, because that number came from
  sampled human populations. Ratios, orderings and directions survive a domain
  move; absolute magnitudes need calibrating against the body being read
- `carve_audit.SYSTEM_REGISTRY` ships empty. The boundary audit is a harness for
  documentation work that has not been done, and returns `INSUFFICIENT_DATA`

### Prompt evaluator (`hands_lie_detector.prompt`)
- Sends the full rubric as a structured prompt to vision LLMs
- Parses JSON scores from model responses
- Feeds parsed scores through `ScoreEvaluator` for consistent interpretation
- Supports Anthropic (Claude) and OpenAI (GPT-4o) APIs

## Conventions

- Scoring rubrics use bullet-point ranges (e.g., "0-5:", "6-15:", "16-25:")
- Documentation tone: direct, practical, slightly irreverent
- Core principle: **structural adaptation over surface appearance**
- Method: before tuning anything, ask what the functional form cannot represent
  at ANY parameter setting. If that is the quantity in question, it is a
  replacement, not an edit. Three instances are indexed in `gated-not-summed.md`
- Method: run the strip first. Render every category noun into force,
  displacement, cycles and duty cycle before reasoning about it. A noun that
  will not render is a ledger class and must not be weighted
- Python: type hints, dataclasses, Python 3.11+

## Training Data Sources

The vision classifier needs labeled hand images. Here are practical paths to build a dataset.

### Bootstrap with the prompt evaluator (recommended first step)
Use `PromptEvaluator` to auto-label images from any source below. This produces 7-category scores directly compatible with the training pipeline. Human review is still needed — the whole point of this project is that models get this wrong — but it gives a starting baseline to iterate from.

### Public image datasets
- **11k Hands Dataset** — 11,000+ hand images with demographic metadata (dorsal/palmar). Academic dataset from Mahmoud Afifi. Good volume, but mostly clean studio shots — useful as low-score training examples.
- **EgoHands** — 4,800 hand images from first-person video (Indiana University). Hands in natural contexts, varied activities.
- **Oxford Hand Dataset** — Hand detection/pose dataset. Less relevant for scoring but usable as negative/baseline examples.

### Community sources (requires permission/scraping ethics)
- **Reddit** — Subreddits like r/BlueCollarWomen, r/tradesman, r/Carpentry, r/MechanicAdvice, r/Welding, r/Gardening frequently have hand photos in real work contexts. Also r/hands and r/mildlyinteresting for variety.
- **Flickr Creative Commons** — Search "working hands", "farmer hands", "mechanic hands", "carpenter hands". Filter by CC license. Wide variety of real-world shots.
- **Wikimedia Commons** — Category:Hands has thousands of freely licensed images across contexts.

### Manual collection
- **Photograph known workers** — Most reliable ground truth. Photograph hands of people with known occupations/experience levels, then score with the rubric. Even 50-100 well-labeled images is enough to start fine-tuning.
- **Before/after washing** — Photograph the same hands dirty and clean. This directly tests the core thesis and creates paired training data.

### Crowdsourcing
- **Amazon Mechanical Turk / Prolific** — Post a task: "Photograph your hands (clean, palms up and down)" with a survey about occupation, years of manual work, and trade type. This gives images + self-reported ground truth.
- **University studies** — Partner with an occupational health or ergonomics lab that already collects hand data.

### Synthetic / augmentation
- **Cross-label with the rubric** — Have multiple human raters score the same images independently, then average. Reduces individual bias.
- **Augmentation** — The dataset module already includes random crop, flip, and color jitter. For hands specifically, consider adding: random rotation (hands are photographed at all angles), brightness variation (indoor vs outdoor), and slight perspective warps.

### Data format
Place images in `data/images/` and create `data/labels.csv`:
```csv
filename,texture_persistence,wear_localization,micro_injury_history,tendon_vein_definition,nail_evidence,symmetry_of_wear,climate_ppe
001.jpg,18,14,11,10,7,8,4
002.jpg,3,2,1,3,2,1,0
```

### Labeling workflow suggestion
1. Collect raw images from sources above
2. Run `PromptEvaluator` to generate initial labels
3. Human-review and correct scores using the rubric from `scoring-metrics.md`
4. Train the vision classifier
5. Compare classifier vs prompt evaluator vs human scores to find disagreements
6. Focus labeling effort on disagreement cases

## Known Issues

- `scoring-metrics.md` has duplicate content (lines 1-108 and 112-208)
- Future expansions in README (clean_but_used, callus_memory, etc.) not yet developed
- Vision classifier needs labeled training data to be useful — no dataset included yet
- Four documented sign errors in the thickness/injury scale, none fixed:
  saturation reads as expertise, acute damage reads as incompetence, thickness
  baseline uncalibrated, residual mark uncalibrated. See `band-not-scale.md`
- Four documented form errors, none fixed in the shipped scale, because each fix
  is a replacement rather than an edit: additive scoring, the monotone thickness
  scale, summed layer weighting, and deposit/draw treated as parallel inputs when
  they carry opposite signs. See `gated-not-summed.md` and `wear-taxonomy.md`
- The band thresholds and the tier-2 items need raking light. No frame in the
  specimen series has it, so tier 2 has never actually been measured. The fix is
  five photographs and one lamp — see `capture-protocol.md`; position 5 is the
  one that matters and nothing substitutes for it
- Specimen 008 rules out SOFT and saturated with high confidence, which is the
  first state resolution in the series. The boundary map is still unresolved
- No counterface has been recorded for any wear observation — no tool handles.
  One boot is on record (specimen 007) and it falsifies one category prediction;
  `CATEGORY_PREDICTIONS` holds exactly one entry
- The sole audit has no denominator either, but unlike every other gap in this
  repo it is cheap to build: any worn boot with a stated job title is a data
  point
- The attribution-retrofit account is **model-authored testimony**, not a
  measured phenomenon. Three of its four tests also depend on scoring
  instruments with no operationalization and no inter-rater agreement, so their
  deltas are marks rather than measurements — `results_are_interpretable`
  returns False and the report says so above its own numbers. Only the
  no-destination test is runnable today: binary, no scale, its own control
- **Provenance convention for documents.** A claim originating in a model's
  account of its own behavior is `RECONSTRUCTED` and must carry a header saying
  so. Do not let a model's story about itself accumulate as a finding
- Scoring rubric and vision heads are additive; they cannot represent the
  integration term. See `reference-class-empty.md` for what this costs and for
  the specific directional error in `WEAR_LOCALIZATION` (multi-domain wear reads
  as unlocalized and scores toward the cosplayer band)
- H1 vs H2 in `reference-class-empty.md` is open; no discriminator has been run
- The rubric and vision heads are monotone in thickness and rank a saturated
  hand highest. `hands_lie_detector.band` demonstrates the sign error but does
  not fix the rubric — a replacement scale has to be built on contrast from the
  start, and after the band thresholds are measured rather than stipulated
- `no_context_no_props` is now the core test in README and has not been run
- The repo conflates two axes: the perception layer (reading markers) and the
  partition underneath (how load is attributed and weighted). Movement on one is
  not movement on the other. See `specimen-record.md`
- **The calibration standard is undocumented.** Load history, dated band-state
  series, maintenance practice and held-out stimuli are the standard, and they
  exist only in the operator's head. The band series is also the only local
  source for the tissue coefficients the seam refuses to import. The schemas are in code (`LoadBlock`,
  `read_band`, `MANAGEMENT_ACTS`, `commit_stimulus`) and the fields are empty.
  This cannot be authored from inside the repo — writing a plausible load history
  would manufacture the calibration artifact itself
- No cross-section has been recorded. The audit instruments have nothing to
  measure against until held-out stimuli exist
- Arm B's condition coordinates are unwritten (`ConditionSpec` in
  `audit/condition.py`). Same gap as the calibration standard and for the same
  reason: plausible coordinates written from here would manufacture the contrast
  point rather than record it
- Reference photographs supplied in conversation are NOT committed, deliberately.
  Publishing an image spends it as a stimulus — commit hashes with
  `commit_stimulus()` and hold the files outside the repo
- `DEFAULT_BANDS` reports physical scores onto employment status ("Casual
  Hobbyist" at 31-55, below "Working Hands"). See `economic-carve.md` — renaming
  the bands alone is the wrong fix; the physical scale needs somewhere to land
  that isn't a job title
- The labeling workflow above collects "occupation, years of manual work, trade
  type" — enrollment by pay code, which is the unit `economic-carve.md` argues
  is wrong. Unpaid load enters as absent or as "hobby"

## Tests

```bash
python -m unittest discover tests
```

Zero dependencies. The suite is mostly claim tests: each asserts something the
documents state in prose, so the prose cannot drift from the code (no payment
field on `LoadBlock`, residual never attributed to an enrolled domain, defaults
flagged as stipulated, audit ships unrun).

## Git Workflow

- Primary branch: `main`
- Feature branches prefixed with `claude/`
- Commits are descriptive, using "Create" for new files and "Update" for edits
