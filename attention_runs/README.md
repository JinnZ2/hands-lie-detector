# attention_runs

Logs and results for attention-as-sensor runs. Frames are **not** committed:
backlit frames show the vein pattern and stay on device (see
`attention-as-sensor.md`, Privacy). `.gitignore` keeps image files out of this folder.

State, 2026-11-15:
- `claude_2026-11-15_morning.json`: Claude's log for the morning photos. It is valid.
  - 4 entries, in order:
    1. sun / low sun glare through windshield
    2. hand acting as light filter (transmission through tissue)
    3. orange-red rim along finger edges
    4. web between fingers (thin tissue glow)
  - Provenance: CARRIED, from the chat transcript (a reply to 2 hand photos sent with a greeting only).
  - `before_task_prompt` is true, set by the operator: photos and greeting only, no task.
  - Regions are null until drawn.
  - Excluded: palm creases and weathered texture. They were attended only after the "hands don't lie" cue, once a task had been given. They are recorded in the log's provenance, not as entries.
    - The validator refuses a log with `before_task_prompt` false. A separate log of them would therefore not run as a probe log.
  - Open: the reply covered **2 photos**, and a log is per image. `image_sha256` is null until the log is bound to one frame, or split per frame.
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
