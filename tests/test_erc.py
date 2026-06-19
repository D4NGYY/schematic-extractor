from __future__ import annotations

import networkx as nx

from src.core.erc import ERCReport, ERCSeverity, ERCViolation, run_erc
from src.core.graph_builder import ComponentNode, NetNode, PinNode
from src.ml.clustering import ComponentCluster

# ── helpers ──────────────────────────────────────────────────────────────────

def _cluster(cid: int = 0) -> ComponentCluster:
    return ComponentCluster(
        cluster_id=cid, segments=[], shapes=[], text_blocks=[],
        bbox=(0.0, 0.0, 10.0, 10.0), center=(5.0, 5.0),
    )


def _comp(node_id: str, class_name: str = "resistor") -> ComponentNode:
    return ComponentNode(
        node_id=node_id, ref=node_id, class_name=class_name,
        cluster=_cluster(),
    )


def _net(net_id: str, name: str | None = "VCC") -> NetNode:
    return NetNode(net_id=net_id, name=name)


def _build_clean_graph() -> tuple[nx.Graph, dict[str, ComponentNode], dict[str, NetNode]]:
    """Grafo pulito: R1 ─ N1 ─ C1, tutti connessi, net nominata."""
    g = nx.Graph()
    r1 = _comp("R1")
    r1.pins = [PinNode(pin_id="R1_1", connected_net="N1")]
    c1 = _comp("C1", "capacitor")
    c1.pins = [PinNode(pin_id="C1_1", connected_net="N1")]
    n1 = _net("N1", "VCC")

    g.add_node("R1", bipartite=0)
    g.add_node("C1", bipartite=0)
    g.add_node("N1", bipartite=1)
    g.add_edge("R1", "N1")
    g.add_edge("C1", "N1")

    return g, {"R1": r1, "C1": c1}, {"N1": n1}


# ── clean graph ───────────────────────────────────────────────────────────────

class TestERCCleanGraph:
    def test_no_violations_on_clean_graph(self) -> None:
        """Grafo corretto → zero violazioni."""
        g, comps, nets = _build_clean_graph()
        report = run_erc(g, comps, nets)
        assert report.violations == [], report.violations

    def test_summary_zero(self) -> None:
        g, comps, nets = _build_clean_graph()
        report = run_erc(g, comps, nets)
        assert report.summary() == "0 error(s), 0 warning(s)"

    def test_has_no_errors(self) -> None:
        g, comps, nets = _build_clean_graph()
        assert not run_erc(g, comps, nets).has_errors()


# ── isolated component ────────────────────────────────────────────────────────

class TestIsolatedComponent:
    def test_isolated_component_is_error(self) -> None:
        """Componente senza edge → ISOLATED_COMPONENT error."""
        g = nx.Graph()
        r1 = _comp("R1")
        g.add_node("R1", bipartite=0)

        report = run_erc(g, {"R1": r1}, {})
        isolated = [v for v in report.violations if v.rule == "ISOLATED_COMPONENT"]
        assert len(isolated) == 1
        assert isolated[0].severity == ERCSeverity.ERROR
        assert isolated[0].node_id == "R1"

    def test_not_isolated_when_connected(self) -> None:
        g, comps, nets = _build_clean_graph()
        report = run_erc(g, comps, nets)
        assert not any(v.rule == "ISOLATED_COMPONENT" for v in report.violations)


# ── floating pin ──────────────────────────────────────────────────────────────

class TestFloatingPin:
    def test_floating_pin_is_warning(self) -> None:
        """Pin con connected_net=None → FLOATING_PIN warning."""
        g = nx.Graph()
        r1 = _comp("R1")
        r1.pins = [PinNode(pin_id="R1_1", connected_net=None)]
        g.add_node("R1", bipartite=0)

        report = run_erc(g, {"R1": r1}, {})
        floating = [v for v in report.violations if v.rule == "FLOATING_PIN"]
        assert len(floating) == 1
        assert floating[0].severity == ERCSeverity.WARNING

    def test_no_floating_pins_when_all_connected(self) -> None:
        g, comps, nets = _build_clean_graph()
        report = run_erc(g, comps, nets)
        assert not any(v.rule == "FLOATING_PIN" for v in report.violations)


# ── dangling net ──────────────────────────────────────────────────────────────

class TestDanglingNet:
    def test_dangling_net_is_warning(self) -> None:
        """Net con un solo componente collegato → DANGLING_NET warning."""
        g = nx.Graph()
        r1 = _comp("R1")
        r1.pins = [PinNode(pin_id="R1_1", connected_net="N1")]
        n1 = _net("N1")
        g.add_node("R1", bipartite=0)
        g.add_node("N1", bipartite=1)
        g.add_edge("R1", "N1")

        report = run_erc(g, {"R1": r1}, {"N1": n1})
        dangling = [v for v in report.violations if v.rule == "DANGLING_NET"]
        assert len(dangling) == 1
        assert dangling[0].severity == ERCSeverity.WARNING
        assert dangling[0].node_id == "N1"

    def test_unconnected_net_is_error(self) -> None:
        """Net senza nessun componente → UNCONNECTED_NET error."""
        g = nx.Graph()
        n1 = _net("N1")
        g.add_node("N1", bipartite=1)

        report = run_erc(g, {}, {"N1": n1})
        unconnected = [v for v in report.violations if v.rule == "UNCONNECTED_NET"]
        assert len(unconnected) == 1
        assert unconnected[0].severity == ERCSeverity.ERROR

    def test_no_dangling_when_two_components(self) -> None:
        g, comps, nets = _build_clean_graph()
        report = run_erc(g, comps, nets)
        assert not any(v.rule in ("DANGLING_NET", "UNCONNECTED_NET") for v in report.violations)


# ── unnamed net ───────────────────────────────────────────────────────────────

class TestUnnamedNet:
    def test_unnamed_net_is_warning(self) -> None:
        """Net con name=None → UNNAMED_NET warning."""
        g = nx.Graph()
        n1 = _net("N1", name=None)
        g.add_node("N1", bipartite=1)

        report = run_erc(g, {}, {"N1": n1})
        unnamed = [v for v in report.violations if v.rule == "UNNAMED_NET"]
        assert len(unnamed) == 1
        assert unnamed[0].severity == ERCSeverity.WARNING

    def test_named_net_no_unnamed_violation(self) -> None:
        g, comps, nets = _build_clean_graph()
        report = run_erc(g, comps, nets)
        assert not any(v.rule == "UNNAMED_NET" for v in report.violations)


# ── ERCReport helpers ─────────────────────────────────────────────────────────

class TestERCReport:
    def test_errors_and_warnings_partitioned(self) -> None:
        report = ERCReport(violations=[
            ERCViolation(ERCSeverity.ERROR, "R1", "err"),
            ERCViolation(ERCSeverity.WARNING, "W1", "warn"),
        ])
        assert len(report.errors) == 1
        assert len(report.warnings) == 1

    def test_has_errors_true_when_errors_present(self) -> None:
        report = ERCReport(violations=[
            ERCViolation(ERCSeverity.ERROR, "R1", "err"),
        ])
        assert report.has_errors()

    def test_has_errors_false_when_only_warnings(self) -> None:
        report = ERCReport(violations=[
            ERCViolation(ERCSeverity.WARNING, "W1", "warn"),
        ])
        assert not report.has_errors()

    def test_summary_format(self) -> None:
        report = ERCReport(violations=[
            ERCViolation(ERCSeverity.ERROR, "R", "e"),
            ERCViolation(ERCSeverity.ERROR, "R", "e"),
            ERCViolation(ERCSeverity.WARNING, "W", "w"),
        ])
        assert report.summary() == "2 error(s), 1 warning(s)"
