"""Frame spec contract for parallel export."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _norm_spec_for_cache(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        # Stable float key (avoids worker float noise in keys)
        return round(float(value), 9)
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return ("ndarray", arr.dtype.str, tuple(int(x) for x in arr.shape), arr.tobytes())
    if isinstance(value, dict):
        return tuple(sorted((str(k), _norm_spec_for_cache(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_norm_spec_for_cache(v) for v in value)
    return repr(value)


def spec_content_key(spec: dict[str, Any] | "FrameSpec") -> str:
    """Hash of render-relevant fields only (excludes timeline index)."""
    if isinstance(spec, FrameSpec):
        payload = (spec.kind, _norm_spec_for_cache(spec.params))
    else:
        # Drop index if present — it is timeline order, not pixel identity
        items = {k: v for k, v in spec.items() if k != "index"}
        payload = tuple(sorted((str(k), _norm_spec_for_cache(v)) for k, v in items.items()))
    return hashlib.blake2b(repr(payload).encode(), digest_size=16).hexdigest()


def spec_render_key(spec: dict[str, Any] | "FrameSpec") -> str:
    """Backward-compatible alias: content key (not index-dependent)."""
    return spec_content_key(spec)


@dataclass(frozen=True)
class FrameSpec:
    index: int
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def content_key(self) -> str:
        return spec_content_key(self)

    def render_key(self) -> str:
        return self.content_key()

    def to_dict(self) -> dict[str, Any]:
        # index kept for debugging; renderers should ignore it for pixels
        return {"index": self.index, "kind": self.kind, **self.params}
