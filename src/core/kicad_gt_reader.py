from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KicadPin:
    ref: str
    pin_num: str
    x: float
    y: float


@dataclass
class KicadSymbol:
    ref: str
    lib_id: str
    x: float
    y: float
    pins: list[KicadPin] = field(default_factory=list)


@dataclass
class KicadWire:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class KicadJunction:
    x: float
    y: float


@dataclass
class KicadLabel:
    text: str
    x: float
    y: float
    is_global: bool


@dataclass
class KicadSchematic:
    symbols: list[KicadSymbol] = field(default_factory=list)
    wires: list[KicadWire] = field(default_factory=list)
    junctions: list[KicadJunction] = field(default_factory=list)
    labels: list[KicadLabel] = field(default_factory=list)


@dataclass
class GTGraph:
    components: set[str] = field(default_factory=set)
    # net_id -> set of (ref, pin_num)
    nets: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    # The actual number of isolated nets formed by wires/labels
    net_count: int = 0


def tokenize_sexpr(text: str) -> list[str]:
    """Semplice tokenizer per s-expressions."""
    pattern = r'\(|\)|"[^"]*"|[^\s()]+'
    return [match.group(0) for match in re.finditer(pattern, text)]


def parse_sexpr(tokens: list[str]) -> Any:
    """Parser ricorsivo per s-expressions."""
    if not tokens:
        return []
    token = tokens.pop(0)
    if token == '(':
        lst = []
        while tokens and tokens[0] != ')':
            lst.append(parse_sexpr(tokens))
        if tokens:
            tokens.pop(0)  # pop ')'
        return lst
    elif token == ')':
        raise ValueError("Unexpected ')'")
    else:
        if token.startswith('"') and token.endswith('"'):
            return token[1:-1]
        return token


def _find_all(lst: Any, name: str) -> list[list[Any]]:
    """Trova tutte le s-exp che iniziano con `name` nel primo livello di lst."""
    if not isinstance(lst, list):
        return []
    results = []
    for item in lst:
        if isinstance(item, list) and item and item[0] == name:
            results.append(item)
    return results


def _find_first(lst: Any, name: str) -> list[Any] | None:
    for item in lst:
        if isinstance(item, list) and item and item[0] == name:
            return item
    return None


def parse_kicad_sch(path: Path | str) -> KicadSchematic:
    """Estrae gli elementi ground truth da un file .kicad_sch."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    tokens = tokenize_sexpr(text)
    ast = parse_sexpr(tokens)

    if not ast or not isinstance(ast, list) or ast[0] != "kicad_sch":
        raise ValueError("File is not a valid kicad_sch")

    sch = KicadSchematic()

    # 1. Parse lib_symbols to extract pin positions relative to symbol origin
    lib_symbols_node = _find_first(ast, "lib_symbols")
    lib_pins = {} # lib_id -> list of (pin_num, x, y, rot)
    if lib_symbols_node:
        for sym in _find_all(lib_symbols_node, "symbol"):
            if len(sym) < 2:
                continue
            lib_id = sym[1]
            pins = []

            def extract_pins_recursive(node: Any) -> None:
                for child in node:
                    if isinstance(child, list) and child:
                        if child[0] == "pin":
                            at_node = _find_first(child, "at")
                            num_node = _find_first(child, "number")
                            if at_node and num_node:
                                px = float(at_node[1])
                                py = float(at_node[2])
                                rot = float(at_node[3]) if len(at_node) > 3 else 0.0
                                pin_num = str(num_node[1])
                                pins.append((pin_num, px, py, rot))
                        elif child[0] == "symbol":
                            extract_pins_recursive(child)

            extract_pins_recursive(sym)
            if pins:
                lib_pins[lib_id] = pins

    # Extract wires
    for w in _find_all(ast, "wire"):
        pts_node = _find_first(w, "pts")
        if pts_node:
            xys = _find_all(pts_node, "xy")
            if len(xys) >= 2:
                x1, y1 = float(xys[0][1]), float(xys[0][2])
                x2, y2 = float(xys[1][1]), float(xys[1][2])
                sch.wires.append(KicadWire(x1, y1, x2, y2))

    # Extract junctions
    for j in _find_all(ast, "junction"):
        at_node = _find_first(j, "at")
        if at_node:
            x, y = float(at_node[1]), float(at_node[2])
            sch.junctions.append(KicadJunction(x, y))

    # Extract local and global labels
    for lbl_type in ["label", "global_label"]:
        for lbl in _find_all(ast, lbl_type):
            text_val = lbl[1]
            at_node = _find_first(lbl, "at")
            if at_node:
                x, y = float(at_node[1]), float(at_node[2])
                sch.labels.append(
                    KicadLabel(
                        text=text_val,
                        x=x,
                        y=y,
                        is_global=(lbl_type == "global_label")
                    )
                )

    # Extract symbols
    import math
    for sym in _find_all(ast, "symbol"):
        lib_id_node = _find_first(sym, "lib_id")
        at_node = _find_first(sym, "at")
        mirror_node = _find_first(sym, "mirror")

        lib_id = lib_id_node[1] if lib_id_node else ""
        x = float(at_node[1]) if at_node else 0.0
        y = float(at_node[2]) if at_node else 0.0
        rot = float(at_node[3]) if (at_node and len(at_node) > 3) else 0.0

        mirror_x = False
        mirror_y = False
        if mirror_node and len(mirror_node) > 1:
            if mirror_node[1] == "x": mirror_x = True
            if mirror_node[1] == "y": mirror_y = True

        ref = ""
        for prop in _find_all(sym, "property"):
            if len(prop) > 2 and prop[1] == "Reference":
                ref = prop[2]
                break

        symbol = KicadSymbol(ref=ref, lib_id=lib_id, x=x, y=y)

        # Extract pins
        inline_pins = False
        for pin in _find_all(sym, "pin"):
            pin_num = pin[1]
            p_at = _find_first(pin, "at")
            if p_at:
                inline_pins = True
                px, py = float(p_at[1]), float(p_at[2])
                symbol.pins.append(KicadPin(ref=ref, pin_num=pin_num, x=px, y=py))

        # If no inline pins, look up lib_pins and transform
        if not inline_pins and lib_id in lib_pins:
            angle_rad = math.radians(rot)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            for pin_num, px, py, prot in lib_pins[lib_id]:
                # 1. Mirror
                mx = -px if mirror_x else px
                my = -py if mirror_y else py
                # 2. Rotate (KiCad Y axis is down)
                rx = mx * cos_a - my * sin_a
                ry = mx * sin_a + my * cos_a
                # 3. Translate
                abs_x = x + rx
                abs_y = y + ry
                symbol.pins.append(KicadPin(ref=ref, pin_num=pin_num, x=abs_x, y=abs_y))

        sch.symbols.append(symbol)

    return sch


def build_gt_graph(sch: KicadSchematic) -> GTGraph:
    """Costruisce il grafo ground truth (connessioni ideali) dal KicadSchematic."""
    graph = GTGraph()

    for sym in sch.symbols:
        # KiCad refs prefixed with '#' (#PWR power ports, #FLG power flags) are
        # virtual symbols / net anchors, not physical components — exclude them
        # from the component set (their pins still anchor nets below).
        if sym.ref.startswith("#"):
            continue
        graph.components.add(sym.ref)

    # Group connected points
    # Two elements connect if they share exactly the same (x,y) coordinate.
    # To handle floating point, use round to 3 decimals.
    from collections import defaultdict
    adj: dict[tuple[float, float], set[tuple[float, float]]] = defaultdict(set)

    for w in sch.wires:
        p1 = (round(w.x1, 3), round(w.y1, 3))
        p2 = (round(w.x2, 3), round(w.y2, 3))
        adj[p1].add(p2)
        adj[p2].add(p1)

    # Note: T-junctions in KiCad are typically split into multiple wire segments
    # that share a common endpoint. Or there's a junction point.
    # If a junction point falls ON a wire, we split the wire.
    for j in sch.junctions:
        jx, jy = round(j.x, 3), round(j.y, 3)
        for w in sch.wires:
            # Check if (jx, jy) is strictly inside the wire segment
            p1 = (round(w.x1, 3), round(w.y1, 3))
            p2 = (round(w.x2, 3), round(w.y2, 3))
            if p1 == (jx, jy) or p2 == (jx, jy):
                continue

            # Distance check to see if J is on the segment P1-P2
            import math
            L = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            if L < 1e-5:
                continue
            d1 = math.hypot(jx-p1[0], jy-p1[1])
            d2 = math.hypot(jx-p2[0], jy-p2[1])
            if abs(d1 + d2 - L) < 1e-3:
                # J is on the wire, so it connects them
                adj[p1].add((jx, jy))
                adj[(jx, jy)].add(p1)
                adj[p2].add((jx, jy))
                adj[(jx, jy)].add(p2)
                # Remove the original p1-p2 connection
                if p2 in adj[p1]: adj[p1].remove(p2)
                if p1 in adj[p2]: adj[p2].remove(p1)

    # Find connected components (nets)
    visited = set()
    net_groups = []

    for node in list(adj.keys()):
        if node not in visited:
            stack = [node]
            group = set()
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    group.add(curr)
                    stack.extend(adj[curr])
            net_groups.append(group)

    # For points without wires (e.g. isolated label or pin)
    # We will just map them directly to a point group.

    # Map each (x,y) to a net_id
    pt_to_net: dict[tuple[float, float], str] = {}
    for i, group in enumerate(net_groups):
        net_id = f"GT_NET_{i+1}"
        for pt in group:
            pt_to_net[pt] = net_id

    # Now associate pins and labels
    net_labels: dict[str, str] = {}
    next_net_idx = len(net_groups) + 1

    def get_net(pt: tuple[float, float]) -> str:
        nonlocal next_net_idx
        if pt not in pt_to_net:
            pt_to_net[pt] = f"GT_NET_{next_net_idx}"
            next_net_idx += 1
        return pt_to_net[pt]

    for lbl in sch.labels:
        pt = (round(lbl.x, 3), round(lbl.y, 3))
        nid = get_net(pt)
        net_labels[nid] = lbl.text

    # Link pins to nets
    for sym in sch.symbols:
        for pin in sym.pins:
            pt = (round(pin.x, 3), round(pin.y, 3))
            nid = get_net(pt)
            graph.nets.setdefault(nid, set()).add((sym.ref, pin.pin_num))

    # Rename nets to their label if they have one
    renamed_nets: dict[str, set[tuple[str, str]]] = {}
    for nid, pins in graph.nets.items():
        name = net_labels.get(nid, nid)
        # Avoid naming collision
        if name in renamed_nets:
            renamed_nets[name].update(pins)
        else:
            renamed_nets[name] = pins

    graph.nets = renamed_nets
    graph.net_count = len(set(pt_to_net.values()))

    return graph
