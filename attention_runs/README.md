# attention_runs

Logs and results for attention-as-sensor runs. Frames are **not** committed:
backlit frames show the vein pattern and stay on device (see
`attention-as-sensor.md`, Privacy). `.gitignore` keeps image files out of this folder.

State, 2026-11-15:
- Claude's morning log is split into one log per photo, with the same 4 entries in each:
  - `claude_2026-11-15_morning_photo1.json`
  - `claude_2026-11-15_morning_photo2.json`
  - Entries, in order:
    1. sun / low sun glare through windshield
    2. hand acting as light filter (transmission through tissue)
    3. orange-red rim along finger edges
    4. web between fingers (thin tissue glow)
  - `log_kind` free, `before_task_prompt` true (photos and greeting only, set by the operator).
  - `per_photo_order` UNRESOLVED: "reply covered the pair; order not separable by photo".
    A run passes this through and prints it beside the shuffled-order control.
  - Provenance CARRIED, from the chat transcript. Regions null until drawn.
  - `image_sha256` null until each photo's PNG arrives. Which uploaded frame is
    photo1 and which is photo2 is not recorded.
- Prompted logs: `claude_2026-11-15_morning_photo1_prompted.json` and `..._photo2_prompted.json`.
  - 2 entries: palm creases, weathered texture. They were attended only after the
    "hands don't lie" cue.
  - `log_kind` prompted, `before_task_prompt` false. Stored and validated for later
    comparison with the free log; **never used as a probe**. `run`, `sweep` and
    `ObserverLogs` refuse them.
  - Check a stored log of either kind: `python3 -m hands_lie_detector.attention.check FILE [--size W H]`.
- First real run: **pending**. No PNG has arrived, and no regions are drawn.

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
