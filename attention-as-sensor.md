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
| 1 | `log.py` | One log per observer per image. A log is refused unless `before_task_prompt` is true. Entries carry a rank (1..n in listing order), what was noticed, and a region `[x0, y0, x1, y1]` or `null` (unmapped, counted, never placed). Logs are **never merged**: `ObserverLogs` refuses a second log for an observer it holds, and no function combines entries across observers. Observer slots (`claude`, `other_models`, `human`) are null until a log fills them. |
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
