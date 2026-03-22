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
│   └── prompt/                   # Requires: anthropic or openai
│       └── evaluator.py          # Vision LLM scoring with rubric prompt
├── scoring-metrics.md            # Main 100-point hand scoring rubric (v0.1)
├── feet-lie-detector.md          # Parallel scoring system for feet
├── gloves-debunk.md              # Addresses the "I wore gloves" excuse
├── known_failure_cases.md        # Catalogues specific vision model failures
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
- **feet-lie-detector.md** — Analogous rubric for feet.
- **gloves-debunk.md** — Why gloves don't erase structural adaptation.
- **known_failure_cases.md** — Specific vision model failure modes.

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

### Prompt evaluator (`hands_lie_detector.prompt`)
- Sends the full rubric as a structured prompt to vision LLMs
- Parses JSON scores from model responses
- Feeds parsed scores through `ScoreEvaluator` for consistent interpretation
- Supports Anthropic (Claude) and OpenAI (GPT-4o) APIs

## Conventions

- Scoring rubrics use bullet-point ranges (e.g., "0-5:", "6-15:", "16-25:")
- Documentation tone: direct, practical, slightly irreverent
- Core principle: **structural adaptation over surface appearance**
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

## Git Workflow

- Primary branch: `main`
- Feature branches prefixed with `claude/`
- Commits are descriptive, using "Create" for new files and "Update" for edits
