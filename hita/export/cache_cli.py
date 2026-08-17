"""python -m hita.export.cache — GC / size helpers."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HITA durable cache utilities")
    parser.add_argument("command", choices=["gc", "size"], help="gc or size")
    parser.add_argument("--chapter", type=int, default=None)
    parser.add_argument("--keep-series", default=None)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)

    from hita.export.cache import cache_size_bytes, gc_cache

    root = args.root
    if args.command == "size":
        print(cache_size_bytes(args.chapter, root))
        return 0
    result = gc_cache(chapter=args.chapter, keep_series=args.keep_series, root=root)
    print(f"removed={result['removed']} bytes_freed={result['bytes_freed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
