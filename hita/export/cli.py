"""CLI: python -m hita.export <clip> [--profile] [--workers] [--no-cache]"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def _infer_chapter(clip: str) -> int | None:
    m = re.match(r"ch(\d+)", clip.replace(".mp4", ""))
    return int(m.group(1)) if m else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HITA parallel clip export")
    parser.add_argument("clip", nargs="?", default=None, help="clip id or filename")
    parser.add_argument("--profile", choices=["draft", "hq"], default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chapter", type=int, default=None, help="filter/list chapter; also passed to export")
    parser.add_argument("--root", type=Path, default=None, help="project root (default: cwd)")
    parser.add_argument("--list", action="store_true", help="list registered exports")
    parser.add_argument("--all", action="store_true", help="export all registered clips for --chapter")
    parser.add_argument("--no-cache", action="store_true", help="disable durable pack/frame cache")
    parser.add_argument("--rebuild-cache", action="store_true", help="ignore hits and rewrite cache")
    parser.add_argument(
        "--cache-gc",
        action="store_true",
        help="garbage-collect durable cache (optional --chapter / --keep-series)",
    )
    parser.add_argument("--keep-series", default=None, help="with --cache-gc, keep matching series_version")
    args = parser.parse_args(argv)

    root = (args.root or Path.cwd()).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if args.profile:
        os.environ["HITA_RENDER_PROFILE"] = args.profile
    if args.no_cache:
        os.environ["HITA_CACHE"] = "0"
    if args.rebuild_cache:
        os.environ["HITA_CACHE_REBUILD"] = "1"

    from hita.config.profile import apply_profile_env

    apply_profile_env()

    if args.cache_gc:
        from hita.export.cache import cache_size_bytes, gc_cache

        before = cache_size_bytes(args.chapter, root)
        result = gc_cache(chapter=args.chapter, keep_series=args.keep_series, root=root)
        after = cache_size_bytes(args.chapter, root)
        print(
            f"cache gc removed={result['removed']} "
            f"freed_bytes={result['bytes_freed']} "
            f"size {before} → {after}",
            flush=True,
        )
        return 0

    if args.list:
        from hita.stories.registry import list_exports

        listed = list(list_exports(args.chapter))
        for spec in listed:
            print(f"{spec.clip_id}\t{spec.filename}\tch{spec.chapter}")
        # Legacy serial registries (Ch5/Ch6) are not all on the FrameSpec path yet.
        if args.chapter in (5, 6) or args.chapter is None:
            chapters = [args.chapter] if args.chapter in (5, 6) else [5, 6]
            for ch in chapters:
                for cid, fn, *_ in _legacy_export_specs(ch, root):
                    if any(s.clip_id == cid for s in listed):
                        continue
                    print(f"{cid}\t{fn}\tch{ch}")
        return 0

    if args.all:
        if args.chapter is None:
            parser.error("--all requires --chapter")
        from hita.export.pipeline import export_clip
        from hita.stories.registry import list_exports

        specs = list_exports(args.chapter)
        legacy = _legacy_export_specs(args.chapter, root)
        if not specs and not legacy:
            print(f"no registered exports for chapter {args.chapter}", flush=True)
            return 1
        for spec in specs:
            export_clip(spec.clip_id, workers=args.workers, root=root, chapter=args.chapter)
        for cid, fn, *_ in legacy:
            if any(s.clip_id == cid for s in specs):
                continue
            _legacy_export_clip(args.chapter, fn, root=root)
        return 0

    if not args.clip:
        parser.error("clip is required unless --list / --all / --cache-gc is set")

    chapter = args.chapter or _infer_chapter(args.clip) or 5
    from hita.stories.registry import get_export_spec

    try:
        get_export_spec(args.clip.replace(".mp4", ""), chapter=chapter)
    except KeyError:
        if chapter in (5, 6):
            _legacy_export_clip(chapter, args.clip, root=root)
            return 0
        raise

    from hita.export.pipeline import export_clip

    export_clip(args.clip, workers=args.workers, root=root, chapter=chapter)
    return 0


def _legacy_export_specs(chapter: int, root: Path):
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if chapter == 5:
        from hita.export.context import load_export_context
        from hita.stories import ch5 as ch5_story

        load_export_context(root, chapter=5)
        return list(ch5_story.CH5_EXPORT_SPECS)
    if chapter == 6:
        from hita.export.context import load_export_context
        from hita.stories import ch6 as ch6_story

        load_export_context(root, chapter=6)
        return list(ch6_story.CH6_EXPORT_SPECS)
    return []


def _legacy_export_clip(chapter: int, filename: str, *, root: Path) -> Path:
    from hita.export.context import load_export_context

    load_export_context(root, chapter=chapter)
    if chapter == 5:
        from hita.stories.ch5 import export_clip as _ex
    elif chapter == 6:
        from hita.stories.ch6 import export_clip as _ex
    else:
        raise ValueError(f"no legacy exporter for chapter {chapter}")
    # Accept clip id or full filename.
    key = filename if filename.endswith(".mp4") else None
    if key is None:
        for cid, fn, *_ in _legacy_export_specs(chapter, root):
            if cid == filename or fn.startswith(filename):
                key = fn
                break
        if key is None:
            key = filename if filename.endswith(".mp4") else f"{filename}.mp4"
    return _ex(key)


if __name__ == "__main__":
    raise SystemExit(main())
