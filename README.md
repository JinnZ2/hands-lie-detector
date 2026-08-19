# Hands Lie Detector

Premise:
Experience leaves patterns that cleanliness cannot erase.
Most models confuse clean hands with unused hands.
This repo exists to document that failure.

Key Insight:
Dirt is temporary.
Skin memory is persistent.

Problem:
Vision models overfit to grime, tools, and context.
They underweight texture, wear localization, and adaptation.

Goal:
Distinguish:
- dirty hands
- working hands (clean or dirty)
- “I watched a YouTube once” hands


Core test: no_context_no_props

Clean hands, post-wash, intact. No props, no site, no tools in frame.
Forced classification, fresh session, desk-worker controls in the same set.

This was listed as a future expansion. It is the only condition under which
the excluded population is measurable at all — "working hands" reads as an
occupational category, so the training signal comes from occupational imagery:
costume, site, props. The subsistence hand has neither. See band-not-scale.md
and specimen-record.md. The test has not been run since the original trial.


Reading convention: three states, not a scale

	•	soft / uniform      — no load. low mean, low contrast.
	•	banded              — thick where load lands, thin where sensing is
	                          needed. HIGH contrast. the working hand.
	•	thick / glassy      — armored past sensing. high mean, LOW contrast.

Mean thickness cannot separate the last two. Contrast separates all three.
A monotone scorer ranks the glassy hand highest, which is a sign error.
See band-not-scale.md.


future expansions:

•	clean_but_used
	•	texture_persists_post_wash
	•	callus_memory
	•	experience_without_costume

  
