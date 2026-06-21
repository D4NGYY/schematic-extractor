"""Qualitative error analysis: WHERE does net-F1 loss come from, per board.

Read-only. For each board: build the graph, run the same greedy net-mapping the
metric uses, then break the false-negatives (missed GT net memberships) down by:
  - power vs signal net (GND/VCC/+/PWR... vs Net-NN),
  - GT net degree (how many GT refs the net touches) -> high-degree miss = a
    rail the pipeline fragmented; low-degree miss = a short signal wire dropped,
  - component class of the ref that lost the membership.
Tells us the dominant failure mode instead of optimizing blind.

Run: PYTHONHASHSEED=0 PYTHONPATH=. python diagnosi_d3/error_analysis.py [board ...]
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.kicad_gt_reader import build_gt_graph, parse_kicad_sch
from src.core.logging_config import configure_logging
from src.core.pdf_parser import VectorExtractor

configure_logging(log_level="CRITICAL")
BASE = Path("test_input/multi_schematic")
POWER = ("GND", "VCC", "VDD", "VSS", "VEE", "PWR", "+", "VBUS", "GROUND")


def is_power(net_id: str) -> bool:
    u = str(net_id).upper()
    return any(p in u for p in POWER)


def analyze(name: str) -> None:
    d = BASE / name
    pages = VectorExtractor().extract(str(next(d.glob("*.pdf"))))
    gt = build_gt_graph(parse_kicad_sch(next(d.glob("*.kicad_sch"))))
    ext_cn: dict[str, set] = defaultdict(set)
    cls: dict[str, str] = {}
    for pi, pg in enumerate(pages):
        b = BipartiteGraphBuilder(cluster_eps=None)
        b.build_from_page(pg)
        valid = set(b.nets)
        for ref, c in b.components.items():
            cls[ref] = c.class_name
            for p in c.pins:
                if p.connected_net in valid:
                    ext_cn[ref].add(f"{pi}:{p.connected_net}")
    gt_cn: dict[str, set] = defaultdict(set)
    gt_deg: dict[str, int] = {}
    for nid, pins in gt.nets.items():
        gt_deg[nid] = len({r for r, _ in pins})
        for ref, _ in pins:
            gt_cn[ref].add(nid)
    common = set(ext_cn) & set(gt.components)

    # greedy mapping (same as metric)
    enr: dict = defaultdict(set)
    gnr: dict = defaultdict(set)
    for ref, nets in ext_cn.items():
        if ref in common:
            for n in nets:
                enr[n].add(ref)
    for ref, nets in gt_cn.items():
        if ref in common:
            for n in nets:
                gnr[n].add(ref)
    sc = sorted(((len(er & gr), en, gn) for en, er in enr.items()
                 for gn, gr in gnr.items() if er & gr), reverse=True)
    nm: dict = {}
    ue: set = set()
    ug: set = set()
    for _o, en, gn in sc:
        if en not in ue and gn not in ug:
            nm[en] = gn
            ue.add(en)
            ug.add(gn)

    tp = fp = fn = 0
    fn_power = fn_signal = 0
    fn_by_deg = Counter()
    fn_by_class = Counter()
    fp_by_class = Counter()
    for ref in common:
        e = ext_cn[ref]
        g = gt_cn[ref]
        mapped = {nm.get(n) for n in e if n in nm}
        for gn in g:
            if gn in mapped:
                tp += 1
            else:
                fn += 1
                if is_power(gn):
                    fn_power += 1
                else:
                    fn_signal += 1
                deg = gt_deg.get(gn, 1)
                fn_by_deg["deg>=5" if deg >= 5 else ("deg3-4" if deg >= 3 else "deg1-2")] += 1
                fn_by_class[cls.get(ref, "?")] += 1
        for en in e:
            if nm.get(en) not in g:
                fp += 1
                fp_by_class[cls.get(ref, "?")] += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"\n=== {name}  F1={f1:.3f}  (tp={tp} fp={fp} fn={fn}, overlap_refs={len(common)}/{len(gt.components)}) ===")
    print(f"  FN split:  power={fn_power}  signal={fn_signal}")
    print(f"  FN by GT-net degree: {dict(fn_by_deg)}")
    print(f"  FN by component class (top): {dict(fn_by_class.most_common(5))}")
    print(f"  FP by component class (top): {dict(fp_by_class.most_common(5))}")


def main() -> None:
    names = sys.argv[1:] or ["sallen_key", "ecc83-pp", "ampli_ht", "arduino_micro"]
    for n in names:
        try:
            analyze(n)
        except Exception as e:  # noqa: BLE001
            print(f"{n}: ERROR {e}")


if __name__ == "__main__":
    main()
