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
- Prompted logs: `claude_2026-11-15_morning_photo1_prompted.json`, `..._photo2_prompted.json`.
  - 2 entries: palm creases, weathered texture, attended only after the "hands don't lie" cue.
  - `log_kind` prompted, `before_task_prompt` false. Stored for later comparison with
    the free logs; **never used as a probe** (`run`, `sweep` and `ObserverLogs` refuse them).
- Road set: no logs yet. Any log for road1 or road2 must be prompted.
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
