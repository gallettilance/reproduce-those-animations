"""Canonical Ch1 survey dataset (Scene 1–8) — matches notebook cell 2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dataset2D:
    study: np.ndarray
    exam: np.ndarray
    y: np.ndarray
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    name: str = "ch1_survey"

    @property
    def diff(self) -> np.ndarray:
        return self.study - self.exam


def ch1_survey_dataset() -> Dataset2D:
    separable_points = [
        (2, 3, 0),
        (4, 5, 0),
        (5, 6, 0),
        (1, 3, 0),
        (2, 4, 0),
        (4, 6, 0),
        (1, 4, 0),
        (3, 6, 0),
        (1, 6, 0),
        (3, 2, 1),
        (5, 4, 1),
        (6, 5, 1),
        (4, 2, 1),
        (6, 4, 1),
        (3, 1, 1),
        (4, 1, 1),
        (5, 2, 1),
        (6, 3, 1),
        (6, 2, 1),
        (6, 1, 1),
    ]
    noisy_symmetric_points = [
        (2, 1, 0),
        (1, 2, 1),
        (3, 4, 1),
        (4, 3, 0),
        (3, 5, 1),
        (5, 3, 0),
    ]
    pts = separable_points + noisy_symmetric_points
    study = np.asarray([p[0] for p in pts], dtype=float)
    exam = np.asarray([p[1] for p in pts], dtype=float)
    y = np.asarray([p[2] for p in pts], dtype=int)
    return Dataset2D(study=study, exam=exam, y=y, xlim=(0.0, 7.0), ylim=(0.0, 7.0), name="ch1_survey")
