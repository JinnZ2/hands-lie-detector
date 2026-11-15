# Related fields — entheseal changes (bioarchaeology)

Status: **PROPOSED**, co-authored. This is a pointer list, not a literature
review.

Each entry has a verification state:
- **VERIFIED (abstract)**: the bibliographic record and abstract were read in
  Europe PMC on 2026-11-15, and the quoted lines are from that abstract;
- **UNVERIFIED**: the record was not found or not reachable from the session
  that wrote this, so the entry is carried as delivered.

No full text was read for any entry.

---

## 1. Entheseal changes (EC), bioarchaeology since the 1980s

- **Delivered note:** a bone-level deposit; the same move as tissue, applied
  to dead subjects.
- **State: UNVERIFIED (carried).** The field and its dating are carried as
  delivered. No source was read for "since the 1980s".

## 2. Karakostis et al. 2017

- **Record:** Karakostis FA, Hotz G, Scherf H, Wahl J, Harvati K.
  "Occupational manual activity is reflected on the patterns among hand
  entheses." *American Journal of Physical Anthropology* 164:30–40.
  doi:10.1002/ajpa.23253. PMID 28542729.
- **State: VERIFIED (abstract).**
- **Delivered TODO — what the two patterns are — resolved at abstract depth.**
  The abstract proposes the two patterns as "two synergistic muscle groups".
  - "Individuals involved in intense manual labor during their lifetime
    presented a distinctive pattern of hand entheses, consistent with the
    application of high grip force."
  - "By contrast, individuals with less strenuous and/or highly mechanized
    occupations showed an entheseal pattern related to the thumb intrinsic
    muscles."
- **Method, from the abstract:**
  - 45 male adults from mid-19th-century Basel;
  - occupations "clearly documented and could be evaluated according to
    historical sources";
  - 3D quantification and principal component analysis;
  - checked against age-at-death, body mass and bone length.
- **DERIVED, for this repo:** their ground truth is a recorded occupation
  category, from historical sources. That is the enrollment unit
  `economic-carve.md` argues against. Their groups are "intense manual labor"
  versus "less strenuous and/or highly mechanized", which is a category, not
  a load history. The bone pattern is a deposit, as here. The labels it was
  read against are a ledger, not a deposit. Read before borrowing their
  groups.
- **PROPOSED prediction (co-authored):** continuous multi-domain load, where
  grip-force work and thumb-precision work run together over years, would
  produce an **intermediate or mixed** entheseal pattern, sitting between the
  two Karakostis patterns rather than on either.
  - It is a bone-level prediction. Entheses are not visible in a photograph
    of a living hand, so no frame in this repo can test it.
  - It would need skeletal material whose load history is documented as
    multi-domain. That is the same enrollment gap `reference-class-empty.md`
    describes.
  - Not tested. Karakostis's abstract reports two patterns and does not say
    whether intermediate cases were seen.

## 3. Davis et al. 2012, *International Journal of Osteoarchaeology*

- **Title as delivered:** "Patterns of Interobserver Error in Scoring of EC".
- **Delivered note:** a reader-variance confound, parallel to observer pull.
- **State: UNVERIFIED.** Europe PMC does not return this record; the journal
  is not indexed there. Author list, volume, pages and DOI are not confirmed
  here.
- **Provenance (as delivered):** cited in the reference list of the Wiley
  special issue "Entheseal Changes and Occupation".
- **Verification route:** a title search on Wiley Online Library. Not run
  from this session, because that host is not on its network allowlist.
  Stays UNVERIFIED until a record is found.

## 4. Drapeau 2008, *HOMO* 59:93–109

- **Record:** Drapeau MS. "Enthesis bilateral asymmetry in humans and African
  apes." *HOMO* 59:93–109. doi:10.1016/j.jchb.2007.12.004. PMID 18396287.
- **State: VERIFIED (abstract).**
- **Delivered note:** an asymmetry metric that maps to the left/right photo
  set.
- From the abstract:
  - "the assumption that entheses reflect muscle activity levels has been
    challenged."
  - "This study suggests that enthesis morphology provides information on
    muscle activity levels, with greater development of entheses associated
    with more habitual or powerful muscle use."
- **DERIVED:**
  - The abstract's own caveat (intensity versus frequency of contraction "not
    well understood") is the same open question as this repo's load-mode
    split. A bilateral difference says one side was used more. It does not
    say whether "more" means more often or harder.
  - The same frequency-versus-intensity confound applies to callus edge
    sharpness. Many cycles at a fixed grip geometry and fewer cycles at high
    force could both produce a sharp edge, so a sharp edge alone does not
    say which. Edge sharpness as a discriminator between the two is a
    **CANDIDATE**, not a test. Nothing here shows that the two loads leave
    different edges, and no threshold exists.

## 5. Review: "Entheseal Changes: Benefits, Limitations and Applications in Bioarchaeology"

- **Delivered:** a Pathways review.
- **State: UNVERIFIED.** No record with this title was found in Europe PMC.
  Authors, venue, year and identifier are not confirmed here.
- **Provenance (as delivered):** a ResearchGate PDF and a page on the Pathways
  journal site, under the exact title above.
- **Verification route:** an exact-title search on the Pathways journal site.
  ResearchGate is a secondary copy, not the record. Not run from this session.
  Stays UNVERIFIED until a record is found.

---

## DIFFERENCE TO STATE

```
theirs:    bone, post-mortem, no correction possible
this repo: soft tissue, living, carrier can correct the readout
scope:     tissue gives the integrated SUM; decomposition is still testimony
```

**DERIVED.**
- "No correction possible" is about the subject, not the reader. A dead
  subject cannot dispute a reading. Item 3 still shows readers disagreeing
  with each other, so interobserver error exists on both sides.
- What the living carrier adds is a channel the skeleton lacks: a correction
  from the body's owner. Under `channel-split.md` that correction is CH2
  (annotation), not CH1. It can fix a misread but cannot replace the record.
- The scope line matches `readout-channel.md`. Tissue and bone both give the
  integrated sum, and any split of that sum into domains or occupations is
  testimony. In entheseal work that testimony is the historical occupation
  record (item 2).

## Provenance

| entry | source read | sha256 of record fetched |
|---|---|---|
| 2 | Europe PMC REST search, `DOI:"10.1002/ajpa.23253"`, 2026-11-15T14:58:12Z | `72562f9f44d0cfe8eec3292fcab810ebb63bb33208501939376dd26b11f5da53` |
| 4 | Europe PMC REST search (author + title), 2026-11-15T14:58:35Z | `fa97860cc7299a68a6b47dde436ac4a02beadce54f0830835f374dac273a1175` |
| 3, 5 | Europe PMC REST search returned no matching record | — |

Fetched records are not committed.
