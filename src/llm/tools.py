import re
from typing import Any

import networkx as nx

# --- Semantic enrichment helpers ---------------------------------------------
# Translate raw refs/net names into human-readable descriptions so the LLM can
# answer "what is this circuit" instead of echoing "U0, C5, N6". Backward-
# compatible: tool methods ADD enriched fields alongside the existing ones.

# Ref-designator prefix -> human class (EIA/JEDEC convention, the standard used
# by KiCad and most EDA tools). Used as fallback when the cluster's own
# class_name is "unknown" or missing.
_REF_CLASS: dict[str, str] = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "D": "diode",
    "Q": "transistor",
    "U": "IC",
    "J": "connector",
    "P": "connector",
    "K": "relay",
    "Y": "crystal",
    "F": "fuse",
    "SW": "switch",
    "RV": "potentiometer",
    "TP": "test point",
    "LED": "LED",
    "V": "tube",
    "T": "transformer",
}

# Net names that are power rails (ground / supply). Matched case-insensitively
# against the leading token. Lets the LLM tell rail nets from signal nets.
_RAIL_PATTERNS = re.compile(
    r"^(GND|VCC|VDD|VEE|VSS|AGND|DGND|AVDD|AVCC|"
    r"\+[0-9]+V?|\+[0-9]*\.?[0-9]+V|"
    r"3V3|5V|12V|24V|VBAT|VBUS|VREF|VMEM)\b",
    re.IGNORECASE,
)


def _infer_class_from_ref(ref: str) -> str:
    """Best-effort class from a ref-designator prefix (R->resistor, U->IC, ...).

    Handles multi-letter prefixes (SW, RV, TP, LED) by trying the longest match
    first, falling back to single-letter, then to 'unknown'.
    """
    if not ref:
        return "unknown"
    # Letters-only prefix (drop trailing digits/suffixes).
    m = re.match(r"^([A-Za-z]+)", ref)
    if not m:
        return "unknown"
    letters = m.group(1)
    for length in (len(letters), len(letters) - 1, 1):
        if length <= 0:
            break
        prefix = letters[:length].upper()
        if prefix in _REF_CLASS:
            return _REF_CLASS[prefix]
    return "unknown"


def _component_class(ref: str, data: dict[str, Any]) -> str:
    """Class from the cluster's own class_name, else inferred from the ref prefix."""
    cls = data.get("class_name")
    if cls and cls != "unknown":
        return cls
    return _infer_class_from_ref(ref)


def _is_rail(name: str) -> bool:
    """True if a net name looks like a power rail (GND, VCC, +5V, 3V3, ...)."""
    return bool(_RAIL_PATTERNS.match(name or ""))


def _describe_component(ref: str, data: dict[str, Any]) -> str:
    """One-line human description: 'R1 (resistor, 10k)' or 'U3 (IC)'."""
    cls = _component_class(ref, data)
    val = data.get("value")
    desc = f"{ref} ({cls}"
    if val and val != "None":
        desc += f", {val}"
    return desc + ")"


def _describe_net(name: str, data: dict[str, Any] | None = None) -> str:
    """Net name with type tag: 'GND (power rail)' or 'N6 (signal net)'."""
    kind = "power rail" if _is_rail(name) else "signal net"
    return f"{name} ({kind})"


# --- Schema helpers (unchanged) ----------------------------------------------


def _is_component(data: dict[str, Any]) -> bool:
    """True for component nodes in either schema (test fixtures use type=,
    BipartiteGraphBuilder uses bipartite=0)."""
    return data.get("type") == "component" or data.get("bipartite") == 0


def _is_net(data: dict[str, Any]) -> bool:
    """True for net nodes in either schema (type='net' or bipartite=1)."""
    return data.get("type") == "net" or data.get("bipartite") == 1


def _net_name(data: dict[str, Any], node: Any) -> str:
    """Resolve a human-readable net name across both schemas."""
    return data.get("name") or data.get("net_id") or str(node)


class GraphContext:
    """Wraps the bipartite Components<->Nets networkx graph and exposes query methods."""

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        # Pre-compute some lookups for faster access
        self.components: dict[str, Any] = {}
        self.nets: dict[str, Any] = {}
        for node, data in self.graph.nodes(data=True):
            if _is_component(data):
                # Assumes node_id or ref is the actual name
                ref = data.get("ref", str(node))
                self.components[ref] = {"node_id": node, **data}
            elif _is_net(data):
                # For nets, the node name might be the net_id or name
                name = _net_name(data, node)
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
        connected_nets_detail = []
        connected_components = set()
        connected_components_detail = {}  # ref -> description

        for neighbor in self.graph.neighbors(node_id):
            n_data = self.graph.nodes[neighbor]
            if _is_net(n_data):
                net_name = _net_name(n_data, neighbor)
                connected_nets.append(net_name)
                connected_nets_detail.append(_describe_net(net_name, n_data))
                # Find other components on this net
                for net_neighbor in self.graph.neighbors(neighbor):
                    if net_neighbor != node_id:
                        nn_data = self.graph.nodes[net_neighbor]
                        if _is_component(nn_data):
                            nn_ref = nn_data.get("ref", str(net_neighbor))
                            connected_components.add(nn_ref)
                            if nn_ref not in connected_components_detail:
                                connected_components_detail[nn_ref] = _describe_component(
                                    nn_ref, nn_data
                                )

        return {
            "component": component_ref,
            "component_description": _describe_component(component_ref, self.graph.nodes[node_id]),
            "connected_nets": connected_nets,
            "connected_nets_detail": connected_nets_detail,
            "connected_components": list(connected_components),
            "connected_components_detail": list(connected_components_detail.values()),
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
            if _is_component(n_data):
                # Find which pin is connected to this net (graph_builder uses pin_id)
                edge_data = self.graph.get_edge_data(node_id, neighbor, {})
                pin = edge_data.get("pin_id") or edge_data.get("pin", "unknown")
                ref = n_data.get("ref", str(neighbor))
                components.append({
                    "ref": ref,
                    "pin": pin,
                    "description": _describe_component(ref, n_data),
                })

        return {
            "net": net_name,
            "type": "power rail" if _is_rail(net_name) else "signal net",
            "components": components,
            "num_components": len(components),
        }

    def find_isolated(self) -> dict[str, Any]:
        """Output: {isolated_components, count}"""
        isolated = []
        isolated_detail = []
        for ref, data in self.components.items():
            node_id = data["node_id"]
            if self.graph.degree(node_id) == 0:
                isolated.append(ref)
                isolated_detail.append(_describe_component(ref, data))

        return {
            "isolated_components": isolated,
            "components_detail": isolated_detail,
            "count": len(isolated),
            "note": (
                "Isolated components have no connection to any net. This often means "
                "an extraction gap (symbol not detected) or a genuinely unconnected part."
                if isolated
                else "No isolated components found."
            ),
        }

    def get_component_info(self, component_ref: str) -> dict[str, Any]:
        """Input: ref componente. Output: {ref, value, component_type, bbox, num_pins, connected_nets}"""
        node_id = self._get_comp_node(component_ref)
        if not node_id:
            return {"error": f"Component {component_ref} not found."}

        data = self.graph.nodes[node_id]

        connected_nets = []
        connected_nets_detail = []
        for neighbor in self.graph.neighbors(node_id):
            n_data = self.graph.nodes[neighbor]
            if _is_net(n_data):
                net_name = _net_name(n_data, neighbor)
                connected_nets.append(net_name)
                connected_nets_detail.append(_describe_net(net_name, n_data))

        return {
            "ref": component_ref,
            "description": _describe_component(component_ref, data),
            "value": data.get("value"),
            "component_type": data.get("class_name") or _infer_class_from_ref(component_ref),
            "bbox": data.get("bbox"),
            "num_pins": self.graph.degree(node_id),
            "connected_nets": connected_nets,
            "connected_nets_detail": connected_nets_detail,
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
                matches.append({
                    "ref": ref,
                    "value": val,
                    "description": _describe_component(ref, data),
                })

        return {
            "matches": matches,
            "count": len(matches)
        }

    def get_nets_summary(self, min_components: int = 1) -> dict[str, Any]:
        """Input: min_components (default 1). Output: {nets, count} for nets
        wiring at least min_components components, sorted by component count desc."""
        # Models often pass the threshold as a string ("2"); coerce defensively.
        try:
            min_components = int(min_components)
        except (TypeError, ValueError):
            min_components = 1
        result: list[dict[str, Any]] = []
        for name, data in self.nets.items():
            node_id = data["node_id"]
            neighbors = list(self.graph.neighbors(node_id))
            comps = [
                (nb, self.graph.nodes[nb])
                for nb in neighbors
                if _is_component(self.graph.nodes[nb])
            ]
            if len(comps) >= min_components:
                comp_refs = [d.get("ref", str(nb)) for nb, d in comps]
                result.append({
                    "net": name,
                    "type": "power rail" if _is_rail(name) else "signal net",
                    "num_components": len(comps),
                    "components": comp_refs,
                    "components_detail": [
                        _describe_component(d.get("ref", str(nb)), d) for nb, d in comps
                    ],
                })
        # Sort: power rails first (so the LLM sees structure), then by comp count.
        result.sort(key=lambda x: (x["type"] != "power rail", -x["num_components"]))
        return {"nets": result, "count": len(result)}

TOOLS_SCHEMA: list[dict[str, Any]] = [
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_nets_summary",
            "description": "List nets that wire at least 'min_components' components together, "
                           "sorted by component count. Use for questions about which nets connect "
                           "multiple components.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_components": {"type": "integer", "description": "Minimum number of components a net must connect (default 1)"}
                },
                "required": []
            }
        }
    }
]
