# How I Think About animation toolkit

**Series version:** `0.2.0` (see `hita/config/series_snapshot_v0.json`)

## Layout

```
hita/
  primitives/      # sigmoid, colormap, datasets, icons, knobs
  choreography/    # SigmoidReveal, camera
  stories/         # ch1…ch7 registries (ch7 = native, no notebook exec)
  assets/          # fonts, icons, knobs (numbered + labeled)
  testing/         # golden-frame capture
notebooks/         # thin manifests
renders/
.venv/
```

## Knobs (numbered ↔ param)

```python
from hita.primitives import KnobStyle, draw_knob_row, load_knob_pack

# Numbered faces (1 / 2 / 3)
draw_knob_row(fig, axes_k, ws, we, bb, style=KnobStyle.NUMBERED)

# Param faces (w_ST / w_EL / b)
draw_knob_row(fig, axes_k, ws, we, bb, style=KnobStyle.LABELED, emphasize="all")

# Morph numbered → labeled per slot (Ch4 handoff)
draw_knob_row(fig, axes_k, ws, we, bb, blend=(0.0, 0.5, 1.0))

pack = load_knob_pack(KnobStyle.LABELED)  # or blend=(…)
```

Bundled under `hita/assets/knobs/{numbered,labeled}/`.

## Setup (macOS, Python 3.12)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m ipykernel install --user --name=hita --display-name="HITA (3.12)"
pytest -q
```

## New clips (Ch7+)

See [docs/ADDING_A_CLIP.md](docs/ADDING_A_CLIP.md). Contract:

`build_pack → list[FrameSpec] → render_* → export_clip`

```bash
HITA_RENDER_PROFILE=draft python -m hita.export ch7_01_sigmoid_reveal_demo --workers 4
```


## Env

| Variable | Default | Purpose |
|----------|---------|---------|
| `HITA_RENDER_PROFILE` | `hq` | `draft` \| `hq` |
| `HITA_EXPORT_WORKERS` | cpu−1 (max 8) | Parallel workers |
| `HITA_EXPORT_PARALLEL` | `1` | `0` = serial |
| `HITA_OUTPUT_DIR` | `renders` | Output folder |
