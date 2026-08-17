"""Shared math: sigmoid, logits, smoothstep."""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def logit_plane(w_st: float, w_el: float, b: float, study, exam):
    return float(w_st) * np.asarray(study, dtype=float) + float(w_el) * np.asarray(exam, dtype=float) + float(b)


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, u: float) -> float:
    return float(a) + (float(b) - float(a)) * float(np.clip(u, 0.0, 1.0))
