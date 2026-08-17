"""Golden-frame capture and hashing for visual regression."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from hita.config.series import SeriesConfig

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "docs" / "golden"
MANIFEST_NAME = "manifest.json"


def sha256_png(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_golden(name: str, image: Image.Image, *, golden_dir: Path | None = None) -> Path:
    out_dir = Path(golden_dir or GOLDEN_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    image.save(path)
    return path


def update_manifest(entries: dict[str, str], *, golden_dir: Path | None = None) -> Path:
    out_dir = Path(golden_dir or GOLDEN_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    series = SeriesConfig.load()
    payload = {
        "series_version": series.version,
        "frames": entries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_manifest(*, golden_dir: Path | None = None) -> dict[str, Any]:
    path = Path(golden_dir or GOLDEN_DIR) / MANIFEST_NAME
    return json.loads(path.read_text())


def render_golden_stills() -> dict[str, str]:
    """Render a small set of stills that pin series visual DNA."""
    import os

    os.environ["HITA_RENDER_PROFILE"] = "draft"
    os.environ["HITA_CACHE"] = "0"
    from hita.config.profile import apply_profile_env
    from hita.export.renderers import render_sigmoid_frame
    from hita.stories.ch7 import build_pack_ch7_01

    apply_profile_env()
    pack = build_pack_ch7_01("golden", {})
    # Use coarser mesh for fast deterministic goldens
    from hita.primitives.data_2d import ch1_survey_dataset
    from hita.primitives.sigmoid_3d import SigmoidMesh

    ds = ch1_survey_dataset()
    pack["mesh"] = SigmoidMesh.build(ds.xlim, ds.ylim, n=48)
    pack["dpi"] = 72
    pack["figsize"] = (8.0, 5.0)

    stills = {
        "sigmoid_flat": {"kind": "sigmoid_surface", "morph_u": 0.0, "elev": 26.0, "azim": 25.0, "pass_surface": True},
        "sigmoid_half": {"kind": "sigmoid_surface", "morph_u": 0.5, "elev": 26.0, "azim": 25.0, "pass_surface": True},
        "sigmoid_full": {"kind": "sigmoid_surface", "morph_u": 1.0, "elev": 26.0, "azim": 25.0, "pass_surface": True},
        "sigmoid_topdown": {
            "kind": "sigmoid_surface",
            "morph_u": 1.0,
            "elev": 89.0,
            "azim": 238.0,
            "pass_surface": True,
            "hide_z": True,
            "exam_label_2d": True,
        },
    }
    hashes: dict[str, str] = {}
    for name, spec in stills.items():
        img = render_sigmoid_frame(pack, spec)
        path = write_golden(name, img)
        hashes[name] = sha256_png(path)
    update_manifest(hashes)
    return hashes


if __name__ == "__main__":
    h = render_golden_stills()
    for k, v in h.items():
        print(f"{k}\t{v}")
