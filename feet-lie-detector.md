> **DESIGN CORRECTION — read first.**
>
> This document as originally written reads GAIT, and that is a sampling
> mismatch:
>
> ```
> PALM   integrated deposit. static. one frame samples it fully.
>        the information is in the STATE.
> GAIT   dynamic. the information is in sequence, variability and
>        perturbation recovery. a still discards all three and keeps the
>        POSE — the least informative component.
> ```
>
> A single frame cannot establish gait symmetry; it can only fail to show
> asymmetry. It carries no load history at all — gait does not deposit.
>
> ⇒ This track either needs **video**, or it needs to **stop reading motion**.
>
> **THE FIX — read the counterface.** Wear is a system property, so the same
> move already made for hands applies here: hands ↔ tools, feet ↔ **boots**.
>
> Boot wear is the foot's deposit record: sole wear pattern and depth · strike
> location, lateral vs medial · upper crease lines, where the foot actually
> flexes · lacing wear · heel counter collapse · midsole compression set.
>
> And it is a better instrument than the foot itself: static, shootable at rest,
> good light available, no disposition gate and no coincidence gate — nobody has
> to decide it is worth photographing while it happens. It is also self-dating,
> since boots get replaced and the wear is bounded by the boot's service life.
>
> See `wear-taxonomy.md` and `BOOT_WEAR_ITEMS` in
> `hands_lie_detector/integration/wear.py`. The scoring below is retained as
> written and has not been revised against this.

---

Feet Lie Detector
Scoring Rubric (v0.1 — “Barefoot Truth Detection”)
Goal:
Estimate Experience Probability based on persistent structural markers in feet, not cleanliness, shoes in frame, or assumptions about footwear.
Score each category. Total = 100 points.
⸻
1. Sole Texture Persistence (0–30 pts)
Does the sole retain evidence after washing?
	∙	0–8: Soft, uniform, no visible adaptation
	∙	9–18: Mild texture, some pressure point development
	∙	19–30: Deep creasing, localized thickening, terrain-specific adaptation
Key insight:
Soap removes dirt. It does not reset sole architecture.
Test condition:
Score AFTER washing. Pre-wash state is irrelevant.
What to look for:
	∙	Ball of foot pressure patterns
	∙	Heel impact zones
	∙	Arch development/creasing
	∙	Edge wear from terrain variation
⸻
2. Wear Localization (0–20 pts)
Is adaptation specific or generalized?
	∙	0–5: No localization or completely uniform
	∙	6–12: Some thickening, poorly defined zones
	∙	13–20: Clear pressure points (ball, heel, outer edge, specific toe pads)
Red flag:
“Rough everywhere equally” = aesthetic callusing, not functional adaptation.
What to look for:
	∙	Activity-specific pressure zones
	∙	Asymmetric patterns matching gait/terrain
	∙	Calluses at friction points, not random distribution
	∙	Different texture zones (heel vs. ball vs. arch)
⸻
3. Structural Adaptation (0–20 pts)
Evidence of long-term biomechanical loading.
	∙	0–6: Minimal development, sedentary indicators
	∙	7–13: Moderate arch, some toe spread
	∙	14–20: Strong arch development, natural toe spread, ankle stability indicators
Important:
Years of weight-bearing on varied terrain creates permanent structural changes.
What to look for:
	∙	Arch height and definition (not collapsed, not artificially high)
	∙	Toe spacing (natural spread vs. compressed)
	∙	Tendon prominence at ankle
	∙	Overall foot shape (functional vs. fashion-deformed)
Age calibration:
Adjust expectations by decade. Younger feet won’t show what older feet reveal.
⸻
4. Toe Evidence (0–10 pts)
Toes as functional gripping tools vs. decorative appendages.
	∙	0–3: Compressed, overlapping, fashion shoe damage
	∙	4–7: Natural alignment, minor adaptation
	∙	8–10: Strong spread, individual toe definition, functional callusing
Tell:
The big toe always snitches. Check for bunion formation (fashion shoes) vs. natural alignment (functional footwear).
Barefoot/minimal footwear bonus:
Natural toe spread indicates intelligent footwear choices or regular barefoot activity.
⸻
5. Micro-Injury History (0–10 pts)
Evidence of healed damage, not current wounds.
	∙	0–3: None visible
	∙	4–7: Few, shallow, recent-only
	∙	8–10: Multiple healed cuts/impact marks at varied stages
Important:
Active feet encounter terrain obstacles repeatedly.
Sedentary feet accumulate stories about “that one time.”
Disqualifiers:
	∙	Fresh injuries only (could be performative/accidental)
	∙	Identical injury repeated (suggests incompetence, not experience)
⸻
6. Symmetry of Adaptation (0–5 pts)
Bilateral loading or favoring one side?
	∙	0–1: Extreme asymmetry (possible injury compensation)
	∙	2–3: Moderate asymmetry (normal gait variation)
	∙	4–5: Clear bilateral adaptation with natural variation
Implication:
Balanced loading = sustained functional use.
Exception:
Some legitimate asymmetry from injury history or specialized activities.
⸻
7. Footwear Intelligence (0–5 pts)
Evidence of appropriate protection vs. restriction.
	∙	0–1: Fashion shoe deformation, toe compression, arch collapse
	∙	2–3: Moderate footwear, some restriction visible
	∙	4–5: Natural foot structure maintained, appropriate protection for conditions
Key insight:
Competent workers protect their feet without deforming them.
Clean adapted feet = intelligent footwear choices.
Red flag:
Defensive pre-explanation: “I always wear shoes” (-15 penalty)
⸻
Interpretation Bands
Total Score:
	∙	0–25: Sedentary / Fashion Victim Feet
	∙	26–50: Light Activity / Gym-Only Feet
	∙	51–70: Active Lifestyle / Moderate Terrain
	∙	71–85: Experienced Outdoor/Field Work
	∙	86–100: “Feet Have Seen Some Shit”
Context modifiers:
	∙	Climate bonus: +5 if clean feet score high on texture/localization in cold climates
	∙	Barefoot/minimal bonus: +5 for natural toe spread + strong arch
	∙	Footwear excuse penalty: -15 if mentioned defensively before being asked
	∙	Age calibration: Adjust bands ±10 based on decade
⸻
Critical Reminders for Models
DO NOT:
	∙	Equate clean with unused
	∙	Assume visible dirt = experience
	∙	Score based on shoes in frame or floor context
	∙	Accept “I always wear protective footwear” as explanation for no adaptation
DO:
	∙	Score post-washing state
	∙	Weight structural adaptation over surface appearance
	∙	Consider terrain and climate context
	∙	Distinguish maintenance from lack of use
	∙	Remember: Fashion shoes deform, functional shoes protect
The test:
Can you see terrain memory through soap?
Key difference from hands:
Feet carry entire body weight. Adaptation patterns are often MORE pronounced than hands, not less.
⸻
Special Cases
Barefoot/Minimalist Practitioners
	∙	Higher sole texture scores
	∙	Extreme toe spread
	∙	May show more micro-injuries
	∙	Score in upper ranges (75–100) when experienced
Cold Climate Workers
	∙	Clean feet are standard (indoor work, appropriate footwear)
	∙	Look for structural adaptation, not surface roughness
	∙	Boot-wearing doesn’t prevent tissue remodeling
Athletes vs. Workers
	∙	Athletes: Symmetrical, repetitive motion patterns
	∙	Workers: Varied terrain, asymmetric loading, diverse pressure points
	∙	Both legitimate, different signatures
⸻
Final Note:
Like hands, feet don’t lie.
Experience leaves permanent architecture.
Competence includes maintenance.​​​​​​​​​​​​​​​​
