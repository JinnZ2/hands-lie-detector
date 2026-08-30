# Channel Split — Physical Record and Annotation

**Effective forward only.** Prior repo state keeps its rating. Nothing here
retroactively invalidates an existing specimen.

---

## The problem

The repo has been letting the physical record and the captioning of it run as
one artifact. Those are two channels with **different failure modes**, and
fusing them lets a reader dismiss both by dismissing one.

## The split

| | **CH1 — physical record** | **CH2 — annotation** |
|---|---|---|
| what it is | the measurement itself: callus map, grip topology, joint wear, healed-injury geometry, wear reading | the captioning of CH1 |
| carries a caption? | **no. mute by construction** | it *is* the caption |
| fails by | resolution · occlusion · angle · scale | gameable · memory · framing · vocabulary |
| **cannot** fail by | **lying** — tissue deposits under load and has no access to the categories, so CH1 can be unreadable but not dishonest | **resolution** — a caption is not limited by light angle, which is why it reaches what CH1 cannot, and why it must never stand in for it |

> **Hands don't lie, but hands don't caption.**

The two lists do not overlap. That non-overlap is the whole reason for the
split: a reader who discounts one channel has not touched the other.

## The rule

The channels are **cited** to each other and never merged.

- CH2 may point at CH1. `Annotation.cites` holds a `PhysicalRecord.record_id`,
  and `ChannelSet.add_annotation` refuses an annotation pointing at nothing.
- CH1 may not contain a caption. `PhysicalRecord` has no field for one.
- Citation runs one direction only.

## What the split buys

A claim declares which channel or channels it rests on, and the consequences
follow mechanically:

| support | survives CH2 wrong | survives CH1 unreadable |
|---|---|---|
| CH1 only | **yes** | no |
| CH2 only | no | **yes** |
| both | no | no |

`Claim.falls_with_either` flags the third row, and the flag is the point:

> **A claim resting on both channels is not stronger for it. It is more
> fragile, because it fails if either fails.**

So a fused claim should be split into the part CH1 carries and the part CH2
carries, stated separately. Worked example:

```
claim: the patch sits in the skin plane
  support: ch1_only          survives CH2 wrong: True

claim: the patch is carbon from board work
  support: both              survives CH2 wrong: False
  FUSED: dismissing either channel dismisses the claim.
```

The first survives any error in the account of *how* the patch got there. The
second does not — and previously the repo stated it as one finding.

## Relation to the provenance marks

`audit/specimen.py` already marks lines `MEASURED` / `TESTIMONY` / etc. That is
a **per-line** distinction inside one document. This is a **per-artifact** one,
and the difference matters: a specimen with both mark types interleaved is still
a single artifact that a reader can discard whole.

The marks say what kind of thing each line is. The channel split makes them
separately citable.

## Status

Forward-only. Existing specimen records are not restructured; new records use
`ChannelSet`. Nothing above requires the author in the loop to complete.
