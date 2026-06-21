import networkx as nx
import pytest

from src.llm.tools import GraphContext


@pytest.fixture
def sample_graph() -> nx.Graph:
    g = nx.Graph()
    # Components
    g.add_node("n1", type="component", ref="R1", value="10k", class_name="Resistor")
    g.add_node("n2", type="component", ref="U1", value="LM358", class_name="OpAmp")
    g.add_node("n3", type="component", ref="C1", value="100nF", class_name="Capacitor")
    g.add_node("n4", type="component", ref="R2", value="1k", class_name="Resistor")
    g.add_node("n_iso", type="component", ref="ISO1", value="None", class_name="Unknown")

    # Nets
    g.add_node("net1", type="net", name="VCC")
    g.add_node("net2", type="net", name="GND")
    g.add_node("net3", type="net", name="N1")

    # Edges
    g.add_edge("n1", "net1", pin="1")
    g.add_edge("n1", "net3", pin="2")

    g.add_edge("n2", "net1", pin="8")
    g.add_edge("n2", "net2", pin="4")
    g.add_edge("n2", "net3", pin="3")

    g.add_edge("n3", "net3", pin="1")
    g.add_edge("n3", "net2", pin="2")

    return g

@pytest.fixture
def context(sample_graph: nx.Graph) -> GraphContext:
    return GraphContext(sample_graph)

def test_get_neighbors(context: GraphContext) -> None:
    res = context.get_neighbors("R1")
    assert "error" not in res
    assert res["component"] == "R1"
    assert "VCC" in res["connected_nets"]
    assert "N1" in res["connected_nets"]
    assert "U1" in res["connected_components"]
    assert "C1" in res["connected_components"]

def test_get_path(context: GraphContext) -> None:
    res = context.get_path("R1", "GND") # Wait, GND is a net, the prompt says component to component
    res = context.get_path("R1", "R2")
    assert res["found"] is False

    res = context.get_path("R1", "C1")
    assert res["found"] is True
    assert res["start"] == "R1"
    assert res["end"] == "C1"
    assert "N1" in res["path"] or "VCC" in res["path"]

def test_get_net_components(context: GraphContext) -> None:
    res = context.get_net_components("VCC")
    assert "error" not in res
    assert res["num_components"] == 2
    refs = [c["ref"] for c in res["components"]]
    assert "R1" in refs
    assert "U1" in refs

def test_find_isolated(context: GraphContext) -> None:
    res = context.find_isolated()
    assert res["count"] == 2 # ISO1 and R2 are isolated
    assert "ISO1" in res["isolated_components"]
    assert "R2" in res["isolated_components"]

def test_get_component_info(context: GraphContext) -> None:
    res = context.get_component_info("U1")
    assert "error" not in res
    assert res["ref"] == "U1"
    assert res["value"] == "LM358"
    assert res["num_pins"] == 3
    assert set(res["connected_nets"]) == {"VCC", "GND", "N1"}

def test_search_by_value(context: GraphContext) -> None:
    res = context.search_by_value("10k")
    assert res["count"] == 1
    assert res["matches"][0]["ref"] == "R1"

    res = context.search_by_value(".*F") # Match 100nF
    assert res["count"] == 1
    assert res["matches"][0]["ref"] == "C1"


# --- Real graph schema (graph_builder uses bipartite=0/1 + pin_id, NOT type=/pin=) ---

@pytest.fixture
def real_graph() -> nx.Graph:
    """Mirrors the schema emitted by BipartiteGraphBuilder.build_from_page:
    components carry bipartite=0, nets carry bipartite=1, edges carry pin_id."""
    g = nx.Graph()
    g.add_node("c_r1", bipartite=0, ref="R1", value="10k", class_name="resistor")
    g.add_node("c_u1", bipartite=0, ref="U1", value="LM358", class_name="unknown")
    g.add_node("c_iso", bipartite=0, ref="X1", value="None", class_name="unknown")
    g.add_node("N1", bipartite=1, net_id="N1", name="Net-1")
    g.add_node("N2", bipartite=1, net_id="N2", name="Net-2")
    g.add_edge("c_r1", "N1", pin_id="1")
    g.add_edge("c_u1", "N1", pin_id="3")
    g.add_edge("c_u1", "N2", pin_id="4")
    return g


@pytest.fixture
def real_context(real_graph: nx.Graph) -> GraphContext:
    return GraphContext(real_graph)


def test_real_schema_components_discovered(real_context: GraphContext) -> None:
    # Regression: GraphContext must recognize bipartite=0/1, not only type=.
    assert set(real_context.components.keys()) == {"R1", "U1", "X1"}
    assert "Net-1" in real_context.nets


def test_real_schema_get_neighbors(real_context: GraphContext) -> None:
    res = real_context.get_neighbors("R1")
    assert "error" not in res
    assert "Net-1" in res["connected_nets"]
    assert "U1" in res["connected_components"]


def test_real_schema_get_net_components_reads_pin_id(real_context: GraphContext) -> None:
    res = real_context.get_net_components("Net-1")
    assert res["num_components"] == 2
    pins = {c["ref"]: c["pin"] for c in res["components"]}
    assert pins["R1"] == "1"  # must come from pin_id, not "unknown"


def test_real_schema_find_isolated(real_context: GraphContext) -> None:
    res = real_context.find_isolated()
    assert res["isolated_components"] == ["X1"]
    assert res["count"] == 1


def test_get_nets_summary(real_context: GraphContext) -> None:
    res = real_context.get_nets_summary(min_components=2)
    assert res["count"] == 1
    assert res["nets"][0]["net"] == "Net-1"
    assert res["nets"][0]["num_components"] == 2
    assert set(res["nets"][0]["components"]) == {"R1", "U1"}

    res_all = real_context.get_nets_summary(min_components=1)
    assert res_all["count"] == 2  # Net-1 and Net-2


# --- Semantic enrichment (added to make LLM answers informative) ---
# Backward-compatible: each tool ADDS enriched fields alongside the raw ones.

def test_infer_class_from_ref() -> None:
    from src.llm.tools import _infer_class_from_ref

    assert _infer_class_from_ref("R1") == "resistor"
    assert _infer_class_from_ref("C5") == "capacitor"
    assert _infer_class_from_ref("U3") == "IC"
    assert _infer_class_from_ref("L1") == "inductor"
    assert _infer_class_from_ref("Q2") == "transistor"
    assert _infer_class_from_ref("J1") == "connector"
    assert _infer_class_from_ref("SW1") == "switch"   # multi-letter prefix
    assert _infer_class_from_ref("TP3") == "test point"
    assert _infer_class_from_ref("Z99") == "unknown"  # unrecognized


def test_is_rail_detection() -> None:
    from src.llm.tools import _is_rail

    assert _is_rail("GND") is True
    assert _is_rail("VCC") is True
    assert _is_rail("+5V") is True
    assert _is_rail("3V3") is True
    assert _is_rail("AGND") is True
    assert _is_rail("N6") is False
    assert _is_rail("Net-1") is False
    assert _is_rail("RX") is False


def test_describe_component(real_context: GraphContext) -> None:
    from src.llm.tools import _describe_component

    # Real schema fixture: R1 has class_name="resistor", value="10k".
    data = real_context.graph.nodes["c_r1"]
    assert _describe_component("R1", data) == "R1 (resistor, 10k)"
    # Unknown class + no value -> just ref + inferred class.
    assert _describe_component("R1", {}) == "R1 (resistor)"
    assert _describe_component("U3", {"class_name": "unknown"}) == "U3 (IC)"


def test_get_neighbors_enriched(context: GraphContext) -> None:
    """get_neighbors now returns description fields alongside raw refs."""
    res = context.get_neighbors("R1")
    # Backward-compat: raw fields unchanged.
    assert res["component"] == "R1"
    assert "VCC" in res["connected_nets"]
    # Enriched fields present.
    assert "component_description" in res
    assert "resistor" in res["component_description"].lower()
    assert "connected_nets_detail" in res
    # Power rail should be tagged.
    rail_detail = [d for d in res["connected_nets_detail"] if "VCC" in d]
    assert rail_detail and "power rail" in rail_detail[0]
    signal_detail = [d for d in res["connected_nets_detail"] if "N1" in d]
    assert signal_detail and "signal net" in signal_detail[0]


def test_get_component_info_enriched(context: GraphContext) -> None:
    res = context.get_component_info("U1")
    assert "description" in res
    assert "OpAmp" in res["description"] or "IC" in res["description"]
    # connected_nets_detail tags rail vs signal.
    details = res["connected_nets_detail"]
    assert any("power rail" in d and "VCC" in d for d in details)
    assert any("signal net" in d and "N1" in d for d in details)


def test_get_net_components_enriched(context: GraphContext) -> None:
    # Rail net gets type="power rail".
    res_rail = context.get_net_components("VCC")
    assert res_rail["type"] == "power rail"
    # Each component has a description.
    assert all("description" in c for c in res_rail["components"])
    # Signal net gets type="signal net".
    res_sig = context.get_net_components("N1")
    assert res_sig["type"] == "signal net"


def test_find_isolated_enriched(context: GraphContext) -> None:
    res = context.find_isolated()
    assert "components_detail" in res
    assert "note" in res
    assert len(res["components_detail"]) == res["count"]
    # ISO1 has class_name="Unknown" -> description should infer from ref.
    iso_desc = [d for d in res["components_detail"] if "ISO1" in d]
    assert iso_desc  # present


def test_get_nets_summary_enriched(context: GraphContext) -> None:
    res = context.get_nets_summary(min_components=1)
    # Each net entry has type + components_detail.
    for entry in res["nets"]:
        assert "type" in entry
        assert entry["type"] in ("power rail", "signal net")
        assert "components_detail" in entry
    # Power rails sort first.
    vcc_entry = next(e for e in res["nets"] if e["net"] == "VCC")
    assert vcc_entry["type"] == "power rail"
    # First entry should be a power rail (sorting: rails first).
    assert res["nets"][0]["type"] == "power rail"


def test_search_by_value_enriched(context: GraphContext) -> None:
    res = context.search_by_value("10k")
    assert res["count"] == 1
    match = res["matches"][0]
    assert "description" in match
    assert "resistor" in match["description"].lower()
