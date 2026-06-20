import pytest
import networkx as nx
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
