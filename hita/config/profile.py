"""Unified draft / HQ render profiles."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_PROFILE_ENV = "HITA_RENDER_PROFILE"
_LEGACY_DRAFT_ENV = "CH3_DRAFT_EXPORT"
_LEGACY_PRIOR_FULL_ENV = "CH5_PRIOR_LANDSCAPE_FULL"
_WORKERS_ENV = "HITA_EXPORT_WORKERS"
_LEGACY_WORKERS_ENV = "CH4_EXPORT_WORKERS"
_PARALLEL_ENV = "HITA_EXPORT_PARALLEL"

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "series_snapshot_v0.json"


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in {"1", "true", "yes", "y"}


def active_profile_name() -> str:
    """Resolve profile from env (with legacy shims)."""
    raw = os.environ.get(_PROFILE_ENV, "").strip().lower()
    if raw in {"draft", "hq"}:
        return raw
    if _truthy(os.environ.get(_LEGACY_DRAFT_ENV)):
        return "draft"
    return "hq"


def prior_landscape_full() -> bool:
    """Whether prior-landscape clips use HQ mesh density."""
    if active_profile_name() == "draft":
        return False
    return _truthy(os.environ.get(_LEGACY_PRIOR_FULL_ENV)) or True


@lru_cache(maxsize=1)
def _load_snapshot() -> dict[str, Any]:
    return json.loads(_SNAPSHOT_PATH.read_text())


@dataclass(frozen=True)
class RenderProfile:
    name: str
    export_dpi: int
    anim_dpi: int
    landscape_dpi: int
    sigma_contour_levels: int
    sigma_contour_antialiased: bool
    land_grid: int
    land_grid_coarse: int
    land_grid_fine: int
    land_quadrant_fine: bool
    grid_land_grid: int
    voxel_grid: int
    knob_sweep_n: int
    camera_orbit_n: int
    camera_orbit_deg: float
    ct_sweep_n: int
    ct_pivot_n: int
    voxel_orbit_n: int
    voxel_fill_n: int
    hold_n: int
    ms_per_frame: int
    ch5_grid: int
    ch5_ct_grid: int
    ch5_voxel_grid: int
    ch5_prior_land_full: bool

    @classmethod
    def from_name(cls, name: str) -> RenderProfile:
        snap = _load_snapshot()
        key = name if name in snap["profiles"] else "hq"
        data = snap["profiles"][key]
        return cls(name=key, **data)

    @classmethod
    def draft(cls) -> RenderProfile:
        return cls.from_name("draft")

    @classmethod
    def hq(cls) -> RenderProfile:
        return cls.from_name("hq")

    def frame_count(self, preset: str) -> int:
        snap = _load_snapshot()
        entry = snap["motion_presets"].get(preset, {})
        return int(entry.get(self.name, entry.get("hq", self.hold_n)))

    def grid(self, preset: str) -> int:
        snap = _load_snapshot()
        entry = snap["surface_presets"].get(preset, {})
        return int(entry.get(self.name, entry.get("hq", self.land_grid)))

    def hold(self) -> int:
        return self.hold_n

    def scale_motion(self, hq_n: int, draft_n: int | None = None) -> int:
        """Replace scattered ``_draft_range(hq, draft)`` calls."""
        if self.name == "draft":
            return int(draft_n if draft_n is not None else max(2, hq_n // 8))
        return int(hq_n)

    def is_draft(self) -> bool:
        return self.name == "draft"


def active_profile() -> RenderProfile:
    return RenderProfile.from_name(active_profile_name())


def default_workers() -> int:
    raw = os.environ.get(_WORKERS_ENV, "").strip()
    if not raw:
        raw = os.environ.get(_LEGACY_WORKERS_ENV, "").strip()
    if raw.isdigit():
        return max(int(raw), 1)
    if parallel_enabled() and not _spawn_unsafe():
        n = os.cpu_count() or 4
        return max(min(n - 1, 8), 1)
    return 1


def parallel_enabled() -> bool:
    return os.environ.get(_PARALLEL_ENV, "1").strip() not in {"0", "false", "no"}


def _spawn_unsafe() -> bool:
    """ProcessPoolExecutor spawn fails from Jupyter — common on macOS."""
    import __main__

    mf = getattr(__main__, "__file__", None)
    if not mf:
        return True
    p = Path(str(mf))
    if not p.is_file():
        return True
    if p.suffix in {".ipy", ".ipynb", ""} or "ipykernel" in str(p):
        return True
    return False


def apply_profile_env() -> RenderProfile:
    """Set legacy env vars so existing chapter modules honor the profile."""
    profile = active_profile()
    os.environ["CH3_DRAFT_EXPORT"] = "1" if profile.is_draft() else "0"
    os.environ["CH5_PRIOR_LANDSCAPE_FULL"] = "1" if profile.ch5_prior_land_full else "0"
    return profile
