import json
import re
import networkx as nx
from typing import Any, Dict

class GraphContext:
    """Wraps the bipartite Components<->Nets networkx graph and exposes query methods."""
    
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        # Pre-compute some lookups for faster access
        self.components: dict[str, Any] = {}
        self.nets: dict[str, Any] = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            if node_type == "component":
                # Assumes node_id or ref is the actual name
                ref = data.get("ref", str(node))
                self.components[ref] = {"node_id": node, **data}
            elif node_type == "net":
                # For nets, the node name might be the net_id or name
                name = data.get("name") or str(node)
                self.nets[name] = {"node_id": node, **data}

    def _get_comp_node(self, ref: str) -> str | None:
        """Helper to get the actual graph node ID for a component ref."""
        # Find exact
        if ref in self.components:
            return str(self.components[ref]["node_id"])
        # Case insensitive find
        for k, v in self.components.items():
            if k.upper() == ref.upper():
                return str(v["node_id"])
        return None

    def _get_net_node(self, name: str) -> str | None:
        """Helper to get the actual graph node ID for a net name."""
        if name in self.nets:
            return str(self.nets[name]["node_id"])
        for k, v in self.nets.items():
            if k.upper() == name.upper():
                return str(v["node_id"])
        return None

    def get_neighbors(self, component_ref: str) -> dict[str, Any]:
        """Input: ref componente (es. "R1", "U5"). Output: {component, connected_nets, connected_components}"""
        node_id = self._get_comp_node(component_ref)
        if not node_id:
            return {"error": f"Component {component_ref} not found."}
        
        connected_nets = []
        connected_components = set()
        
        for neighbor in self.graph.neighbors(node_id):
            n_data = self.graph.nodes[neighbor]
            if n_data.get("type") == "net":
                net_name = n_data.get("name") or str(neighbor)
                connected_nets.append(net_name)
                # Find other components on this net
                for net_neighbor in self.graph.neighbors(neighbor):
                    if net_neighbor != node_id:
                        nn_data = self.graph.nodes[net_neighbor]
                        if nn_data.get("type") == "component":
                            nn_ref = nn_data.get("ref", str(net_neighbor))
                            connected_components.add(nn_ref)
                            
        return {
            "component": component_ref,
            "connected_nets": connected_nets,
            "connected_components": list(connected_components)
        }

    def get_path(self, start_ref: str, end_ref: str) -> dict[str, Any]:
        """Input: due ref componente. Output: {start, end, found, path, length}"""
        start_node = self._get_comp_node(start_ref)
        end_node = self._get_comp_node(end_ref)
        
        if not start_node:
            return {"error": f"Start component {start_ref} not found."}
        if not end_node:
            return {"error": f"End component {end_ref} not found."}
            
        try:
            path = nx.shortest_path(self.graph, source=start_node, target=end_node)
            # Map path nodes to readable names
            readable_path = []
            for n in path:
                data = self.graph.nodes[n]
                if data.get("type") == "component":
                    readable_path.append(data.get("ref", str(n)))
                else:
                    readable_path.append(data.get("name", str(n)))
                    
            return {
                "start": start_ref,
                "end": end_ref,
                "found": True,
                "path": readable_path,
                "length": len(path) - 1
            }
        except nx.NetworkXNoPath:
            return {
                "start": start_ref,
                "end": end_ref,
                "found": False,
                "path": [],
                "length": 0
            }

    def get_net_components(self, net_name: str) -> dict[str, Any]:
        """Input: nome net. Output: {net, components, num_components}"""
        node_id = self._get_net_node(net_name)
        if not node_id:
            return {"error": f"Net {net_name} not found."}
            
        components = []
        for neighbor in self.graph.neighbors(node_id):
            n_data = self.graph.nodes[neighbor]
            if n_data.get("type") == "component":
                # Find which pin is connected to this net
                edge_data = self.graph.get_edge_data(node_id, neighbor, {})
                pin = edge_data.get("pin", "unknown")
                components.append({
                    "ref": n_data.get("ref", str(neighbor)),
                    "pin": pin
                })
                
        return {
            "net": net_name,
            "components": components,
            "num_components": len(components)
        }

    def find_isolated(self) -> dict[str, Any]:
        """Output: {isolated_components, count}"""
        isolated = []
        for ref, data in self.components.items():
            node_id = data["node_id"]
            if self.graph.degree(node_id) == 0:
                isolated.append(ref)
                
        return {
            "isolated_components": isolated,
            "count": len(isolated)
        }

    def get_component_info(self, component_ref: str) -> dict[str, Any]:
        """Input: ref componente. Output: {ref, value, component_type, bbox, num_pins, connected_nets}"""
        node_id = self._get_comp_node(component_ref)
        if not node_id:
            return {"error": f"Component {component_ref} not found."}
            
        data = self.graph.nodes[node_id]
        
        connected_nets = []
        for neighbor in self.graph.neighbors(node_id):
            n_data = self.graph.nodes[neighbor]
            if n_data.get("type") == "net":
                connected_nets.append(n_data.get("name", str(neighbor)))
                
        return {
            "ref": component_ref,
            "value": data.get("value"),
            "component_type": data.get("class_name"),
            "bbox": data.get("bbox"),
            "num_pins": self.graph.degree(node_id),
            "connected_nets": connected_nets
        }

    def search_by_value(self, pattern: str) -> dict[str, Any]:
        """Input: pattern regex. Output: {matches, count}"""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return {"error": f"Invalid regex pattern '{pattern}': {e}"}
            
        matches = []
        for ref, data in self.components.items():
            val = data.get("value")
            if val and regex.search(val):
                matches.append({"ref": ref, "value": val})
                
        return {
            "matches": matches,
            "count": len(matches)
        }

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": "Find connected nets and adjacent components for a given component reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "component_ref": {"type": "string", "description": "The component reference (e.g. 'R1', 'U5')"}
                },
                "required": ["component_ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_path",
            "description": "Find the shortest path between two components in the schematic bipartite graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_ref": {"type": "string", "description": "The starting component reference"},
                    "end_ref": {"type": "string", "description": "The target component reference"}
                },
                "required": ["start_ref", "end_ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_net_components",
            "description": "List all components connected to a specific net.",
            "parameters": {
                "type": "object",
                "properties": {
                    "net_name": {"type": "string", "description": "The name of the net (e.g. 'GND', 'VCC', 'N5')"}
                },
                "required": ["net_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_isolated",
            "description": "Find all isolated components (components with 0 connections to any net).",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_component_info",
            "description": "Get detailed metadata for a specific component (type, value, pins, nets).",
            "parameters": {
                "type": "object",
                "properties": {
                    "component_ref": {"type": "string", "description": "The component reference"}
                },
                "required": ["component_ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_value",
            "description": "Search for components by their value using a regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The regex pattern to search for (e.g. '10k', '1K.*')"}
                },
                "required": ["pattern"]
            }
        }
    }
]
