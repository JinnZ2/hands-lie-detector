# Attention as a sensor

Status: **PROPOSED**, co-authored. Built and tested on constructed images only.
Not run on any frame of a hand, by any observer. Nothing below is a finding:
the module produces readings and states their bounds; it does not interpret them.

Code: `hands_lie_detector/attention/`. Tests: `tests/test_attention.py`.

---

## Delivered specification (verbatim)

```
ATTENTION-AS-SENSOR (PROPOSED, co-authored)
purpose: AI attention bias as a calibrated probe; physics
  layer checks what sits under each landing point

1 ATTENTION LOG
  per image, before any task prompt: observer lists what
  it notices, in order → mapped to image regions
  observers: Claude, other models, human (carrier) → separate
  logs; never merged
2 PHYSICS LAYER per region
  luminance gradient · edge strength · specular vs diffuse ·
  transmission (red/blue ratio, backlit frames = path
  opacity per backlit-transmission.md)
3 CORRELATE
  attention landings vs light-mass boundaries
  controls: random regions; shuffled attention order
4 MAPS
  attention + physics signal  → calibrated pointer
  attention, no physics signal → BIAS
  physics signal, no attention → BLIND SPOT
5 DIVERGENCE across observers → atmosphere map
CONSTRAINTS
  stdlib can't decode JPEG/HEIC: accept PNG/PPM (zlib
  decode), or log regions by coordinates; state which
  privacy: backlit frames stay on device (vein rule)
  bounds logged · observer slots null · --small
```

---

## What was built, step by step

| Step | Module | What it does |
|---|---|---|
| input | `image.py` | Decodes **PNG** and **PPM**. PNG must be 8-bit and non-interlaced, any colour type, with zlib, filters 0–4 and CRC checked. PPM must be P6 or P3. Refuses JPEG and HEIC by magic bytes and names the alternative: **coordinates-only** mode (`--size W H`), where regions are logged by pixel coordinates and the physics layer reads UNMEASURED. Every result carries `input_mode` (`decoded:PNG`, `decoded:PPM` or `coordinates_only`). |
| 1 | `log.py` | One log per observer per image. A log is refused unless `before_task_prompt` is true. Entries carry a rank (1..n in listing order), what was noticed, and a region `[x0, y0, x1, y1]` or `null` (unmapped, counted, never placed). Logs are **never merged**: `ObserverLogs` refuses a second log for an observer it holds, and no function combines entries across observers. Observer slots (`claude`, `other_models`, `human`) are null until a log fills them. A second kind, `log_kind` "prompted" (`before_task_prompt` false), is stored and validated (`validate_stored`, `check.py`) for later comparison and is **never** a probe: `validate`, `ObserverLogs`, run and sweep refuse it. Optional `per_photo_order` (RESOLVED / UNRESOLVED) is passed through to the run output. |
| 2 | `physics.py` | Per region, from pixel values only: mean luminance gradient; edge strength (the share of pixels on a light-mass boundary); specular and diffuse shares; transmission as mean R / mean B. Transmission is read only on a frame declared backlit, where it is **path opacity** (`backlit-transmission.md`), and is NOT_APPLICABLE otherwise. |
| 3 | `maps.py` | Per observer, each landing's edge strength. Two seeded controls: the same regions placed at random (statistic: mean edge strength, with the share of random draws that reach it), and the listing order shuffled (statistic: Spearman rank correlation of order against edge strength, with the share of shuffles that reach its absolute value). |
| 4 | `maps.py` | The frame is tiled into cells. Each cell, per observer, is labelled: attended with signal is `CALIBRATED_POINTER`; attended without signal is `BIAS`; signal not attended is `BLIND_SPOT`; neither is `NEITHER`. In coordinates-only mode every cell is `NOT_EVALUABLE`. |
| 5 | `maps.py` | Divergence: for each cell, which observers attended it. Cells attended by some observers and not others are listed with both sets. It is built from the separate readings; no log is merged to make it. |

Run:

```sh
python3 -m hands_lie_detector.attention --image frame.png --log claude.json --log carrier.json
python3 -m hands_lie_detector.attention --size 4032 3024 --log claude.json   # coordinates-only
    [--backlit] [--small] [--grid 8] [--signal-k 1.5] [--seed 0] [--json out.json]
```

A log file:

```json
{"observer": {"kind": "claude", "id": "<model or role>"},
 "image_sha256": "<sha256 of the image file, optional>",
 "before_task_prompt": true,
 "entries": [{"rank": 1, "noticed": "what was noticed", "region": [x0, y0, x1, y1]},
             {"rank": 2, "noticed": "...", "region": null}]}
```

## Choices made in this build (not in the specification)

Each choice is printed in the run's `bounds`.

1. **Luminance** is Y = 0.2126 R + 0.7152 G + 0.0722 B on the encoded 0–255 values, with no gamma linearisation.
2. **Gradient** is the central-difference magnitude of Y. Border pixels carry none.
3. **Light-mass boundary:** a pixel with gradient ≥ `EDGE_T` = 32. Edge strength is the share of a region's pixels above it.
4. **Specular:** Y ≥ 230 with the channel spread (max − min) ≤ 30. **Diffuse:** 20 < Y < 230.
5. **Grid:** 8 × 8 cells by default.
6. **Signal:** a region's edge strength is at least `SIGNAL_K` = 1.5 times the frame's edge fraction. On a frame with no edges at all, any edge counts.
7. **Controls:** 200 draws, seeded (default seed 0).
8. **`--small`:** an integer block-mean downsample until the long side is at most 128 px. Regions are scaled with floor and ceiling so the box keeps covering the same pixels. The dropped remainder rows and columns are stated in `bounds`.
9. **Output refuses to overwrite** an existing `--json` path.

## Bound sweep

`hands_lie_detector/attention/sweep.py` applies the study-aperture Card 08 rule: test every bound the labels rest on, not only the suspected one.

- **Settings.** It runs all 27 combinations of `EDGE_T` (16, 32, 64), `SIGNAL_K` (1.0, 1.5, 2.0) and `GRID` (6, 8, 12). Controls are not run, since they do not set labels.
- **Landings.** Each mapped entry's label is read at every setting and reported as `HOLDS` (one label everywhere) or `SHIFTS`. A shifting landing lists every setting where it differs from the default (32, 1.5, 8). Landings do not depend on `GRID`.
- **Cells.** For each grid, the label counts at every (`EDGE_T`, `SIGNAL_K`) pair, and how many cells hold one label across those 9 settings. Cells are not compared across grids, because the cell geometry changes.
- **Lattice.** A grid-independent view: points on a regular lattice (at most 64 per side, stated) take their cell's label. Reported: how many points hold each label at all 27 settings, and how many shift.

```sh
python3 -m hands_lie_detector.attention.sweep --image frame.png --log claude.json [--log ...] [--backlit] [--small] [--json out.json]
```

The sweep has been run only on constructed images in the tests. A ramp whose gradient sits between `EDGE_T` 16 and 32 is the fixture that must shift, and a flat region is the one that must hold.

## Runs

`attention_runs/` holds logs and results. Frames are never committed (`.gitignore`).

- **First real run: pending.** No PNG and no completed log have arrived.
- **Frame record** `attention_runs/frames.json` (operator labels): photo1 = IMG_4070, photo2 = IMG_4071 (morning pair, free and prompted logs, both backlit, so the physics run stays on device); road1 = IMG_4074, road2 = IMG_4076 (road set, shoulder-mounted, same hand; prompted logs only). `frames.py` checks every log against it: a log's `frame_label` must be in the record and its `log_kind` allowed there.
- **`image_sha256`** is filled only by `frames fill` from a file the decoder decodes, with `image_sha256_source` naming the format (PNG now; JPEG once the decoder handles it). A hash with no source format is refused. All are null today.
- **Claude's logs for the 2026-11-15 morning photos**, one per photo: `attention_runs/claude_2026-11-15_morning_photo1.json` and `..._photo2.json`.
  - The same 4 entries in each, CARRIED from the chat transcript, with `before_task_prompt` true (photos and greeting only).
  - `per_photo_order` UNRESOLVED: the reply covered the pair, so the order is not separable by photo.
  - Regions are null until drawn.
  - Prompted logs (after the task cue; stored, never a probe): photo2 holds palm creases and weathered/dry texture; photo1 holds a 4-item backlit readout (margins and webs glow, palm center and finger-base pads opaque); road1 and road2 are empty with `entries_pending` true until the 4 shoulder photos are labeled. `entries_pending` keeps "not yet assigned" apart from "nothing noticed".
- `python3 -m hands_lie_detector.attention.template` builds a log from an order given as `--item` arguments, with every region null (`--log-kind prompted` for a post-cue log).

## References

`REFERENCES/attention_as_sensor.md`: three pointers, each with its read depth (one PRESS_RELEASE, two SNIPPET; entry 2's authors and venue CARRIED from memory). None is graded above its depth; the relevance note is PROPOSED.

## Privacy

- No pixel array, crop or per-pixel map is ever written. Outputs are region aggregates and the image's sha256.
- A backlit frame shows the vein pattern (rule zero in `backlit-transmission.md`). Backlit frames **stay on device**, and transmission is read only as a region aggregate.
- The test suite checks that no output contains a list on the scale of the frame's pixel count.

## Limits

- All readings are properties of the encoded pixels: exposure, white balance, compression history and the camera pipeline all enter them. None is a property of tissue.
- **Mapping text to a region is the observer's step or the operator's, not this module's.** A log arrives with regions already drawn. Who drew them, and when, belongs in the log's provenance.
- The four cell labels depend on choices 3, 5 and 6. A different threshold moves the boundary between `BIAS` and `CALIBRATED_POINTER`.
- The controls report shares of draws, not a significance test. With few landings, the random-region share is coarse.
- Every test image is constructed. The module has never been run on a photograph of a hand, and no observer log exists yet.
