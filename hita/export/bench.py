"""Benchmark harness: python -m hita.export.bench <clip> [--workers 1,4]"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HITA export timing breakdown")
    parser.add_argument("clip", help="clip id (e.g. ch5_47 or ch7_01)")
    parser.add_argument("--profile", choices=["draft", "hq"], default="draft")
    parser.add_argument("--workers", default="1", help="comma-separated worker counts")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args(argv)

    os.environ["HITA_RENDER_PROFILE"] = args.profile
    if args.no_cache:
        os.environ["HITA_CACHE"] = "0"
    if args.rebuild_cache:
        os.environ["HITA_CACHE_REBUILD"] = "1"

    from hita.config.profile import apply_profile_env, active_profile
    from hita.config.series import SeriesConfig
    from hita.export.cache import cache_enabled, pack_fingerprint
    from hita.export.context import load_export_context, project_root
    from hita.export.pipeline import assemble_frames, render_specs
    from hita.stories.registry import get_export_spec

    apply_profile_env()
    root = (args.root or project_root()).resolve()
    series = SeriesConfig.load()
    spec = get_export_spec(args.clip)
    chapter = int(spec.chapter)
    ctx = load_export_context(root, chapter=chapter)

    print(
        f"bench {spec.clip_id} profile={active_profile().name} "
        f"series={series.version} cache={'on' if cache_enabled() else 'off'}",
        flush=True,
    )

    t_pack0 = time.perf_counter()
    pack = dict(spec.build_pack(spec.clip_id, ctx))
    pack["_hita_series_version"] = series.version
    pack["_hita_clip_id"] = spec.clip_id
    pack_s = time.perf_counter() - t_pack0
    fp = pack_fingerprint(pack)
    print(f"  pack_s={pack_s:.2f} fingerprint={fp[:12]}…", flush=True)

    ctx = dict(ctx)
    ctx["_hita_pack"] = pack
    t_plan0 = time.perf_counter()
    frame_specs = list(spec.builder_plan(spec.clip_id, ctx))
    plan_s = time.perf_counter() - t_plan0
    keys = {s.content_key() for s in frame_specs}
    print(
        f"  plan_s={plan_s:.2f} specs={len(frame_specs)} unique={len(keys)} "
        f"({100.0 * (1.0 - len(keys) / max(len(frame_specs), 1)):.0f}% in-clip)",
        flush=True,
    )

    for w_raw in args.workers.split(","):
        w = int(w_raw.strip())
        t0 = time.perf_counter()
        rendered = render_specs(
            frame_specs,
            pack,
            render_fn_name=spec.render_fn,
            workers=w,
            root=root,
            chapter=chapter,
            progress_label=f"bench_w{w}",
        )
        render_s = time.perf_counter() - t0
        frames = assemble_frames(frame_specs, rendered, dedupe=spec.dedupe_consecutive)
        print(
            f"  workers={w} render_s={render_s:.2f} "
            f"assembled={len(frames)} unique_rendered≈{len(keys)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
