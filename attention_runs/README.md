# attention_runs

Logs and results for attention-as-sensor runs. Frames are **not** committed:
backlit frames show the vein pattern and stay on device (see
`attention-as-sensor.md`, Privacy). `.gitignore` keeps image files out of this folder.

State, 2026-11-15:
- `frames.json` is the frame record (labels from the operator). Images are not committed.

  | label | file | set | log kinds allowed | backlit |
  |---|---|---|---|---|
  | photo1 | IMG_4070 | morning pair | free, prompted | true: physics run stays on device |
  | photo2 | IMG_4071 | morning pair | free, prompted | true: physics run stays on device |
  | road1 | IMG_4074 | road set, shoulder-mounted, same hand | prompted only (post-cue) | not stated |
  | road2 | IMG_4076 | road set, shoulder-mounted, same hand | prompted only (post-cue) | not stated |

  - `image_sha256` is null on every frame. It is filled only by
    `python3 -m hands_lie_detector.attention.frames fill --frame photo1 --image <file>`,
    from a file the decoder decodes (PNG now; JPEG once the decoder handles it).
    `image_sha256_source` logs which format. A refused file gives no hash.
  - `python3 -m hands_lie_detector.attention.frames check` checks every log against the record.
- Claude's morning logs, one per photo, same 4 entries in each:
  - `claude_2026-11-15_morning_photo1.json` (IMG_4070), `claude_2026-11-15_morning_photo2.json` (IMG_4071)
  - Entries, in order:
    1. sun / low sun glare through windshield
    2. hand acting as light filter (transmission through tissue)
    3. orange-red rim along finger edges
    4. web between fingers (thin tissue glow)
  - `log_kind` free, `before_task_prompt` true (photos and greeting only, set by the operator).
  - `per_photo_order` UNRESOLVED: "reply covered the pair; order not separable by photo".
  - Provenance CARRIED, from the chat transcript. Regions null until drawn.
- Prompted logs (all items after the "hands don't lie" cue; source: chat transcript 2026-11-15).
  `log_kind` prompted, `before_task_prompt` false. Stored for later comparison with the
  free logs; **never used as a probe** (`run`, `sweep` and `ObserverLogs` refuse them).
  - `claude_2026-11-15_morning_photo2_prompted.json` (IMG_4071): palm creases (dense,
    thumb base + across palm); weathered/dry texture. Read on photo2: the reply named
    the "second photo".
  - `claude_2026-11-15_morning_photo1_prompted.json` (IMG_4070): backlit readout, in order:
    finger margins glow (thin); finger webs glow (thin); palm center opaque; finger-base
    pads opaque. Creases and texture were removed (the reply said glare washes photo1 out);
    the removal is recorded in the log's provenance.
  - `claude_2026-11-15_road1_prompted.json`, `..._road2_prompted.json`: **empty**,
    `entries_pending` true. The road readout covered 4 shoulder photos (palm, dorsal,
    2 lateral), and which 2 are IMG_4074/IMG_4076 is not in the transcript, so
    `per_photo_order` is UNRESOLVED until labeled. Empty here means "not yet assigned",
    not "nothing noticed".
- First real run: **pending**. No PNG has arrived, and no regions are drawn. Both morning
  frames are backlit, so that run happens on the device holding the frames; only its
  JSON aggregates are committed.

When a PNG and the logs arrive, from the repository root:

```sh
python3 -m hands_lie_detector.attention --image attention_runs/<frame>.png \
    --log attention_runs/<claude log>.json --log attention_runs/<carrier log>.json \
    [--backlit] --json attention_runs/<date>-run.json
python3 -m hands_lie_detector.attention.sweep --image attention_runs/<frame>.png \
    --log ... --json attention_runs/<date>-sweep.json
```

Commit only the JSON results, which are aggregates. Results are readings and
are not interpreted here.
