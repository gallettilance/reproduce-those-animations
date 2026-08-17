"""HITA — How I Think About animation toolkit."""
from __future__ import annotations

import sys
from pathlib import Path

# Chapter modules (ch4_layout, ch5_story, …) live under hita/legacy during migration.
_LEGACY = Path(__file__).resolve().parent / "legacy"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

from hita.chapter.load import ChapterContext, load_chapter
from hita.choreography import SigmoidRevealSequence
from hita.config.profile import RenderProfile, active_profile, apply_profile_env
from hita.export.pipeline import export_clip
from hita.primitives import CMAP, SigmoidMesh, ch1_survey_dataset, sigmoid

__all__ = [
    "CMAP",
    "ChapterContext",
    "RenderProfile",
    "SigmoidMesh",
    "SigmoidRevealSequence",
    "active_profile",
    "apply_profile_env",
    "ch1_survey_dataset",
    "export_clip",
    "load_chapter",
    "sigmoid",
]

__version__ = "0.1.0"
