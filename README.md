# Hands Lie Detector

Premise:
Experience leaves patterns that cleanliness cannot erase.
Most models confuse clean hands with unused hands.
This repo exists to document that failure.

Key Insight:
Dirt is temporary.
Skin memory is persistent.

Problem:
SKIN CLEANLINESS OUTRANKS ALL OTHER CUES — including context that contradicts it.

Not a general context-overfit. A specific dominance ordering, with surface
grime at the top. In the scrubbed specimen the context was present and pointed
correctly — diamond-plate floor, work boots, cable, quick-connect fitting — and
the classification still flipped when the stain came off. See
known_failure_cases.md.

That is a stronger and more falsifiable claim than "models overfit to context,"
and it is derived from a specimen rather than asserted.

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

  
