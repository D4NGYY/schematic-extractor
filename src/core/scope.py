"""Density / scope gate — let the pipeline know when a schematic is beyond its
supported density so it can flag low confidence instead of returning unreliable
topology.

Calibrated on the 32-board KiCad set (net-connectivity F1, hybrid detector):
the catastrophic boards (F1 < 0.5) are EITHER bus-dominated (huge nets-per-
component: rams 65, muxdata 32, interf_u 14, graphic 8.4) OR very large/dense
(electric 97 components, StickHub 94), while every in-scope board (F1 0.7-0.9)
sits at nets/comp <= ~5 and <= ~73 components. So two thresholds cleanly separate
the disasters from the rest:
  * nets_per_component > 8   -> bus-dominated (memory arrays, data buses)
  * num_components    > 80   -> too large / small-symbol density

Note: thresholds were derived from GROUND-TRUTH counts. On the EXTRACTED graph
nets are more fragmented, so the ratio runs higher — treat the gate as advisory
and recalibrate if you wire it to extracted counts in anger.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScopeAssessment:
    """Whether a board is within the supported density envelope."""

    in_scope: bool
    confidence: str  # "high" | "low"
    num_components: int
    num_nets: int
    nets_per_component: float
    reason: str


def assess_scope(
    num_components: int,
    num_nets: int,
    max_nets_per_component: float = 8.0,
    max_components: int = 80,
) -> ScopeAssessment:
    """Classify a board as in/out of the supported density envelope."""
    ratio = (num_nets / num_components) if num_components else float("inf")
    reasons = []
    if ratio > max_nets_per_component:
        reasons.append(
            f"bus-dominated (nets/comp {ratio:.1f} > {max_nets_per_component})"
        )
    if num_components > max_components:
        reasons.append(f"too large ({num_components} comps > {max_components})")
    in_scope = not reasons
    return ScopeAssessment(
        in_scope=in_scope,
        confidence="high" if in_scope else "low",
        num_components=num_components,
        num_nets=num_nets,
        nets_per_component=round(ratio, 2),
        reason="in supported density envelope" if in_scope else "; ".join(reasons),
    )


def assess_builder(builder: object, **kwargs: float) -> ScopeAssessment:
    """Convenience: assess a BipartiteGraphBuilder after build_from_page()."""
    comps = getattr(builder, "components", {})
    nets = getattr(builder, "nets", {})
    return assess_scope(len(comps), len(nets), **kwargs)
