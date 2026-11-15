# attention_runs

Logs and results for attention-as-sensor runs. Frames are **not** committed:
backlit frames show the vein pattern and stay on device (see
`attention-as-sensor.md`, Privacy). `.gitignore` keeps image files out of this folder.

State, 2026-11-15:
- `claude_2026-11-15_morning.template.json`: Claude's log for the morning frame.
  - Provenance: CARRIED, from the chat transcript.
  - Regions are void until drawn.
  - The order text is not yet received, so the template is not yet a valid log.
- First real run: **pending**. No PNG and no completed log have arrived.

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
