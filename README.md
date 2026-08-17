# How I Think About animation toolkit

**Series version:** `0.2.0` (see `hita/config/series_snapshot_v0.json`)

## Layout

```
hita/
  primitives/      # sigmoid, colormap, datasets, icons, knobs
  choreography/    # SigmoidReveal, camera
  stories/         # ch1…ch7 registries + builders/
  legacy/          # ch4–ch6 story/math/layout (migration source)
  assets/          # fonts, icons, knobs (numbered + labeled)
notebooks/         # thin manifests (regenerated from registry)
renders/           # MP4 output + durable cache
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
pip install -e .
python -m ipykernel install --user --name=hita --display-name="HITA (3.12)"
```

## Contributing animations

HITA has two export paths. **New work should use the native path (chapter ≥ 7).** Chapters 5–6 still use a legacy serial builder that is being migrated.

### Export contract (native — preferred)

```
build_pack(clip_id, ctx) → dict        # precompute meshes, datasets, constants
plan(clip_id, ctx)       → list[FrameSpec]   # timeline only — no rendering
render_fn(pack, spec)    → PIL.Image   # one frame (runs in parallel workers)
export_clip              → MP4 in renders/
```

`FrameSpec` is a `(index, kind, params)` record (`hita/export/spec.py`). Params must be picklable and stable across draft/hq profiles.

### Checklist — add a native clip (follow `hita/stories/ch7.py`)

1. **Pack** — `build_pack_ch7_XX(clip_id, ctx)` gathers everything the renderer needs (dataset, mesh, icons, dpi). Use `active_profile()` for draft vs hq sizing.
2. **Plan** — `plan_ch7_XX(clip_id, ctx)` returns `list[FrameSpec]`. Reuse choreography helpers (`hita/choreography/`) or emit specs directly. Scale frame counts with `profile.scale_motion(hq_n, draft_n)` or `profile.frame_count("orbit_360")`.
3. **Render** — point `render_fn` at an existing renderer (`render_sigmoid_frame`, `render_ch5_frame`, …) or add a top-level function in `hita/export/renderers.py` (must be importable in worker processes).
4. **Register** — call `register(ExportSpec(...))` in the chapter story module with `chapter=7`, `requires_frame_spec=True`, `clip_id`, `filename`, `ms_per_frame`, and tags.
5. **Wire up** — append to `CH7_EXPORT_SPECS`, implement `export_clip` + `install`, and import the chapter from `hita/stories/registry.py` `_register_all()`.
6. **Notebook** — regenerate the thin manifest: `python -m hita.notebooks.gen_notebook --chapter 7`.

### Checklist — extend a legacy chapter (Ch5–6, e.g. chapter 6)

Chapter 6 is the most recent large addition. Clips live in `hita/legacy/` and are exposed through a thin shim in `hita/stories/ch6.py`.

1. **Math / layout** — put reusable computation in `hita/legacy/ch6_frequentist.py` (sampling, fits, ellipses) or `hita/legacy/ch6_layout.py` (duo stage, axes). Reuse Ch5 helpers (`ch5_datasets`, `ch5_core`, `ch5_layout`) where the stage matches.
2. **Builder** — in `hita/legacy/ch6_story.py`, add `build_ch6_XX_<slug>(clip_id) -> list[PIL.Image]`. Each function assembles frames (matplotlib → `_fig_to_plot` → `_finish` for the Ch4 duo canvas). Use `_hold(frame, n)` for pauses and `_draft_short(n_hq, n_draft)` for profile-aware counts.
3. **Register** — in `_ch6_build_export_specs()`, call `add("<slug>", build_ch6_XX_<slug>)`. This auto-assigns the next `ch6_NN` id and `ch6_NN_<slug>.mp4` filename.
4. **Shim** — no change needed if you only append to `CH6_EXPORT_SPECS`; `hita/stories/ch6.py` reads the live registry and caches finished MP4s.
5. **Notebook** — `python -m hita.notebooks.gen_notebook --chapter 6`.

For Ch5 clips that are ready for parallel export, also add `plan_*` + `build_pack_*` in `hita/stories/builders/` and register an `ExportSpec` in `hita/stories/registry.py` (see `ch5_47`).

### Preview and export

```bash
# list registered clips for a chapter
python -m hita.export --chapter 6 --list

# fast preview (lower dpi, fewer frames)
HITA_RENDER_PROFILE=draft python -m hita.export ch6_47

# final render, parallel frames where supported
HITA_RENDER_PROFILE=hq python -m hita.export ch7_01_sigmoid_reveal_demo --workers 4

# export every clip in a chapter
HITA_RENDER_PROFILE=draft python -m hita.export --chapter 6 --all
```

Interactive work in a notebook or REPL:

```python
from hita import load_chapter
ctx = load_chapter(6, profile="draft")
export_clip = ctx.globals_dict["export_clip"]
export_clip("ch6_47_avg_grad_ascent_n60.mp4")
```

Outputs land in `renders/` (override with `HITA_OUTPUT_DIR`). Durable pack/frame/clip caches live under `renders/cache/ch<N>/`.


## Env

| Variable | Default | Purpose |
|----------|---------|---------|
| `HITA_RENDER_PROFILE` | `hq` | `draft` \| `hq` |
| `HITA_EXPORT_WORKERS` | cpu−1 (max 8) | Parallel workers |
| `HITA_EXPORT_PARALLEL` | `1` | `0` = serial |
| `HITA_OUTPUT_DIR` | `renders` | Output folder |
