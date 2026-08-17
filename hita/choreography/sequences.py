"""Declarative animation sequences → FrameSpec lists."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from hita.choreography.camera import elev_ramp, orbit_azims
from hita.export.spec import FrameSpec
from hita.primitives.math import smoothstep


@dataclass(frozen=True)
class SigmoidRevealSequence:
    """Ch1 Scene 8 DNA: flat→σ morph, elev tilt to top-down, optional orbit.

    Used by Ch1 clips 74/75 and Ch2 reverse-77 / morph handoffs.
    """

    clip_id: str = "ch1_sigmoid_reveal"
    az0: float = 25.0
    elev0: float = 26.0
    elev_top: float = 89.0
    topdown_az: float = 238.0  # 180 + 58
    topdown_turn_deg: float = 32.0
    morph_n: int = 100
    flat_hold: int = 16
    top_hold: int = 12
    orbit_n: int = 0
    orbit_deg: float = 360.0
    pass_surface: bool = True

    def plan(self) -> list[FrameSpec]:
        specs: list[FrameSpec] = []
        idx = 0

        def add(kind: str, **params: Any) -> None:
            nonlocal idx
            specs.append(
                FrameSpec(
                    index=idx,
                    kind=kind,
                    params={"clip_id": self.clip_id, "kind": kind, "pass_surface": self.pass_surface, **params},
                )
            )
            idx += 1

        for _ in range(max(1, self.flat_hold)):
            add("sigmoid_surface", morph_u=0.0, elev=self.elev0, azim=self.az0)

        for i in range(max(2, self.morph_n)):
            u = i / (self.morph_n - 1) if self.morph_n > 1 else 1.0
            su = smoothstep(u)
            az = self.az0 + (self.topdown_az - self.az0) * su
            add("sigmoid_surface", morph_u=su, elev=self.elev0, azim=az)

        elevs = elev_ramp(self.elev0, self.elev_top, max(2, int(round(self.elev_top - self.elev0)) + 1))
        for elev in elevs:
            add(
                "sigmoid_surface",
                morph_u=1.0,
                elev=float(elev),
                azim=self.topdown_az,
                hide_z=float(elev) >= 87.5,
                exam_label_2d=float(elev) >= 78.0,
            )

        turn_n = max(2, int(round(self.topdown_turn_deg)) + 1)
        for az in np.linspace(self.topdown_az, self.topdown_az + self.topdown_turn_deg, turn_n):
            add(
                "sigmoid_surface",
                morph_u=1.0,
                elev=self.elev_top,
                azim=float(az),
                hide_z=True,
                exam_label_2d=True,
            )

        for _ in range(max(0, self.top_hold)):
            add(
                "sigmoid_surface",
                morph_u=1.0,
                elev=self.elev_top,
                azim=self.topdown_az + self.topdown_turn_deg,
                hide_z=True,
                exam_label_2d=True,
            )

        if self.orbit_n > 0:
            for az in orbit_azims(self.az0, self.orbit_deg, self.orbit_n, endpoint=False):
                add("sigmoid_surface", morph_u=1.0, elev=self.elev0, azim=float(az))

        return specs


def iter_hold(spec: FrameSpec, n: int) -> Iterator[FrameSpec]:
    for i in range(max(0, n)):
        yield FrameSpec(index=spec.index + i, kind=spec.kind, params=dict(spec.params))
