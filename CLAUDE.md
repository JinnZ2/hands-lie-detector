# CLAUDE.md — Hands Lie Detector

## Project Overview

A documentation-only project that defines scoring rubrics for detecting real physical work experience from hand and foot markers. The core thesis: vision models confuse "clean hands" with "unused hands," overfitting to dirt/props/context instead of persistent structural adaptation in skin and tissue.

**No code, no build system, no dependencies.** This repo is pure Markdown documentation.

## Repository Structure

```
hands-lie-detector/
├── README.md                 # Mission statement and project premise
├── scoring-metrics.md        # Main 100-point hand scoring rubric (v0.1)
├── feet-lie-detector.md      # Parallel scoring system for feet
├── gloves-debunk.md          # Addresses the "I wore gloves" excuse
├── known_failure_cases.md    # Catalogues specific vision model failures
├── LICENSE                   # MIT License
└── CLAUDE.md                 # This file
```

## Key Documents

- **scoring-metrics.md** — The primary deliverable. Seven scoring categories (Texture Persistence, Wear Localization, Micro-Injury History, Tendon & Vein Definition, Nail Evidence, Symmetry of Wear, Climate & PPE Intelligence) totaling 100 points. Note: this file contains a duplicate of the rubric starting at ~line 112 (an earlier version without the PPE section).
- **feet-lie-detector.md** — Analogous rubric for feet with similar philosophy.
- **gloves-debunk.md** — Argues that gloves don't erase structural adaptation; addresses the defensive "I wore gloves" claim.
- **known_failure_cases.md** — Documents specific failure modes where models misjudge hand experience.

## Conventions

- All content is Markdown, written in a direct, opinionated tone
- Scoring rubrics use bullet-point ranges (e.g., "0-5:", "6-15:", "16-25:")
- Sections include "Key insight," "Red flag," "Disqualifiers," and "Tell" callouts
- Interpretation bands map total scores to labels (e.g., "0-30: Podcast Hands", "91-100: Don't Ask, Just Hand Them the Tool")
- Context modifiers (climate bonus, glove penalty, age calibration) adjust final scores

## Development Workflow

There is no build, test, or lint step. Contributions are Markdown edits.

When editing:
1. Keep the tone consistent — direct, practical, slightly irreverent
2. Scoring categories should have clear point ranges with observable criteria
3. Include "red flags" and "disqualifiers" to counter common model biases
4. Maintain the core principle: **structural adaptation over surface appearance**
5. Cross-reference related documents where relevant (e.g., nail scoring references glove debunking)

## Known Issues

- `scoring-metrics.md` has duplicate content: the full rubric appears twice (lines 1-108 and lines 112-208). The first copy includes the v0.1 PPE section (Section 7); the second is an older version without it.
- Future expansions listed in README (clean_but_used, texture_persists_post_wash, callus_memory, etc.) are not yet developed.

## Git Workflow

- Primary branch: `main`
- Feature branches prefixed with `claude/`
- Commits are descriptive, using "Create" for new files and "Update" for edits
