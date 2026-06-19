"""Phase 4 — Electrical Rule Check (ERC) sul grafo bipartito Componenti↔Nets."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import networkx as nx
import structlog

from src.core.graph_builder import ComponentNode, NetNode

logger = structlog.get_logger("erc")


class ERCSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ERCViolation:
    severity: ERCSeverity
    rule: str
    message: str
    node_id: str | None = None


@dataclass
class ERCReport:
    violations: list[ERCViolation] = field(default_factory=list)

    @property
    def errors(self) -> list[ERCViolation]:
        return [v for v in self.violations if v.severity == ERCSeverity.ERROR]

    @property
    def warnings(self) -> list[ERCViolation]:
        return [v for v in self.violations if v.severity == ERCSeverity.WARNING]

    def has_errors(self) -> bool:
        return bool(self.errors)

    def summary(self) -> str:
        return f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"


def run_erc(
    graph: nx.Graph,
    components: dict[str, ComponentNode],
    nets: dict[str, NetNode],
) -> ERCReport:
    """Esegue i controlli ERC e ritorna un ERCReport con tutte le violazioni."""
    report = ERCReport()
    _check_isolated_components(graph, components, report)
    _check_floating_pins(components, report)
    _check_dangling_nets(graph, nets, report)
    _check_unnamed_nets(nets, report)
    logger.info(
        "ERC complete",
        errors=len(report.errors),
        warnings=len(report.warnings),
        total=len(report.violations),
    )
    return report


def _check_isolated_components(
    graph: nx.Graph,
    components: dict[str, ComponentNode],
    report: ERCReport,
) -> None:
    """ISOLATED_COMPONENT: componente senza alcuna connessione a net."""
    for node_id, comp in components.items():
        degree = int(graph.degree(node_id)) if node_id in graph else 0
        if degree == 0:
            report.violations.append(ERCViolation(
                severity=ERCSeverity.ERROR,
                rule="ISOLATED_COMPONENT",
                message=(
                    f"Component {node_id!r} ({comp.class_name}) "
                    "has no net connections."
                ),
                node_id=node_id,
            ))


def _check_floating_pins(
    components: dict[str, ComponentNode],
    report: ERCReport,
) -> None:
    """FLOATING_PIN: pin registrato su un componente ma senza net assegnata."""
    for node_id, comp in components.items():
        for pin in comp.pins:
            if pin.connected_net is None:
                report.violations.append(ERCViolation(
                    severity=ERCSeverity.WARNING,
                    rule="FLOATING_PIN",
                    message=(
                        f"Pin {pin.pin_id!r} of component {node_id!r} "
                        "has no net connection."
                    ),
                    node_id=node_id,
                ))


def _check_dangling_nets(
    graph: nx.Graph,
    nets: dict[str, NetNode],
    report: ERCReport,
) -> None:
    """UNCONNECTED_NET (error) o DANGLING_NET (warning) per net con ≤1 connessione."""
    for net_id, net in nets.items():
        degree = int(graph.degree(net_id)) if net_id in graph else 0
        label = net.name or net_id
        if degree == 0:
            report.violations.append(ERCViolation(
                severity=ERCSeverity.ERROR,
                rule="UNCONNECTED_NET",
                message=f"Net {label!r} ({net_id}) is not connected to any component.",
                node_id=net_id,
            ))
        elif degree == 1:
            report.violations.append(ERCViolation(
                severity=ERCSeverity.WARNING,
                rule="DANGLING_NET",
                message=(
                    f"Net {label!r} ({net_id}) connects to only 1 component "
                    "— possible dangling stub."
                ),
                node_id=net_id,
            ))


def _check_unnamed_nets(
    nets: dict[str, NetNode],
    report: ERCReport,
) -> None:
    """UNNAMED_NET: net senza etichetta esplicita."""
    for net_id, net in nets.items():
        if not net.name:
            report.violations.append(ERCViolation(
                severity=ERCSeverity.WARNING,
                rule="UNNAMED_NET",
                message=f"Net {net_id!r} has no label.",
                node_id=net_id,
            ))
