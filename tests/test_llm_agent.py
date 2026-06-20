import networkx as nx
import pytest

from src.llm.agent import MockClient, SchematicAgent
from src.llm.tools import GraphContext


@pytest.fixture
def sample_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("n1", type="component", ref="R1", value="10k", class_name="Resistor")
    g.add_node("net1", type="net", name="N1")
    g.add_node("n2", type="component", ref="U1", value="LM358", class_name="OpAmp")

    g.add_edge("n1", "net1")
    g.add_edge("n2", "net1")
    return g

@pytest.mark.asyncio
async def test_agent_with_mock_client(sample_graph: nx.Graph) -> None:
    context = GraphContext(sample_graph)
    client = MockClient()
    agent = SchematicAgent(context, client)

    # query will hit MockClient, which on call 1 returns a tool_call to get_neighbors("R1")
    # on call 2, it returns "R1 is connected to U1 via net N1."
    response = await agent.query("What is connected to R1?")

    assert "R1 is connected to U1" in response
    assert client.call_count == 2

@pytest.mark.asyncio
async def test_agent_react_fallback(sample_graph: nx.Graph) -> None:
    context = GraphContext(sample_graph)

    class ReActMockClient:
        def __init__(self):
            self.call_count = 0

        async def chat(self, messages, tools=None):
            self.call_count += 1
            class MockMessage:
                def __init__(self, content):
                    self.content = content
                    self.tool_calls = None
            if self.call_count == 1:
                return MockMessage('I need to use a tool. TOOL_CALL: get_neighbors({"component_ref": "R1"})')
            else:
                return MockMessage("ReAct complete.")

    client = ReActMockClient()
    agent = SchematicAgent(context, client)

    response = await agent.query("Test react")
    assert response == "ReAct complete."
    assert client.call_count == 2


# --- ReAct parser robustness (real text shapes emitted by local models) ---

def _agent(sample_graph: nx.Graph) -> SchematicAgent:
    return SchematicAgent(GraphContext(sample_graph), MockClient())


def test_parse_envelope_format(sample_graph: nx.Graph) -> None:
    # qwen Q5 shape: )((( {"name": "...", "arguments": {...}} )))
    txt = ')((( {"name": "get_neighbors", "arguments": {"component_ref": "R1"}} )))'
    name, args = _agent(sample_graph)._parse_react_tool_call(txt)
    assert name == "get_neighbors"
    assert '"component_ref": "R1"' in args


def test_parse_prefixed_object_format(sample_graph: nx.Graph) -> None:
    # qwen Q3 shape: a junk-prefixed name followed by a bare JSON object.
    txt = 'brtc_get_path {"start_ref": "RF1", "end_ref": "R37"}'
    name, args = _agent(sample_graph)._parse_react_tool_call(txt)
    assert name == "get_path"
    assert '"start_ref": "RF1"' in args


def test_parse_positional_format(sample_graph: nx.Graph) -> None:
    # mistral Q3 shape: positional call mapped to schema param order.
    txt = 'get_path("RF1", "R37")'
    name, args = _agent(sample_graph)._parse_react_tool_call(txt)
    assert name == "get_path"
    import json
    assert json.loads(args) == {"start_ref": "RF1", "end_ref": "R37"}


def test_parse_plain_prose_no_false_positive(sample_graph: nx.Graph) -> None:
    # No tool name with call syntax -> must not fabricate a call.
    assert _agent(sample_graph)._parse_react_tool_call("Non ci sono componenti isolati.") is None


def test_execute_tool_rejects_non_dict_args(sample_graph: nx.Graph) -> None:
    # mistral Q2 crash: model passed a bare JSON string, not an object.
    out = _agent(sample_graph)._execute_tool("get_net_components", '"VCC"')
    assert "must be a JSON object" in out
