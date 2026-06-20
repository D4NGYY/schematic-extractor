import pytest
import networkx as nx
from src.llm.tools import GraphContext
from src.llm.agent import SchematicAgent, MockClient

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
