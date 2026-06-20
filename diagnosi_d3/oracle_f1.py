"""ORACLE upper-bound: F1 with a PERFECT component detector, vs the net-tracing
algorithm ceiling. See HANDOFF s19. De-confounded alignment (2026-06-20): the
KiCad->PDF map is a KNOWN similarity (scale 72/25.4 pt/mm); we FIX the scale and
fit only the translation by MEDIAN over name-matched ref anchors (robust to
ref-label float), instead of least-squares-fitting the scale from noisy anchors.

  oracle_f1  : real extracted wire-nets + GT-perfect components/pins  -> segmentation lever
  pure_gt_f1 : our _build_nets on GT wires (mm) + GT pins             -> algorithm ceiling

Run: PYTHONHASHSEED=0 PYTHONPATH=. python diagnosi_d3/oracle_f1.py [board ...]
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.core.graph_builder import BipartiteGraphBuilder, ComponentNode, PinNode
from src.core.kicad_gt_reader import build_gt_graph, parse_kicad_sch
from src.core.pdf_parser import PDFSegment, VectorExtractor
from src.ml.clustering import SpatialClusterer

BASE = Path("test_input/multi_schematic")
MM_TO_PT = 72.0 / 25.4  # exact KiCad mm -> PDF pt scale

def _greedy_f1(ext_cn, gt_cn, common):
    enr=defaultdict(set); gnr=defaultdict(set)
    for ref,nets in ext_cn.items():
        if ref in common:
            for n in nets: enr[n].add(ref)
    for ref,nets in gt_cn.items():
        if ref in common:
            for n in nets: gnr[n].add(ref)
    scores=[(len(er&gr),en,gn) for en,er in enr.items() for gn,gr in gnr.items() if er&gr]
    scores.sort(reverse=True); nm={};ue=set();ug=set()
    for _ov,en,gn in scores:
        if en not in ue and gn not in ug: nm[en]=gn;ue.add(en);ug.add(gn)
    tp=fp=fn=0
    for ref in common:
        e=ext_cn[ref]; g=gt_cn[ref]; mapped={nm.get(n) for n in e if n in nm}
        for gn in g: tp+=1 if gn in mapped else 0; fn+=0 if gn in mapped else 1
        for en in e:
            if nm.get(en) not in g: fp+=1
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
    f1=2*p*r/(p+r) if p+r else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"f1":round(f1,4)}

def _fit_translation(gt_pos, ext_pos, anchors):
    """Fixed scale, median translation; y-sign chosen by lower anchor residual."""
    best=None
    for sy in (MM_TO_PT, -MM_TO_PT):
        tx=float(np.median([ext_pos[k][0]-MM_TO_PT*gt_pos[k][0] for k in anchors]))
        ty=float(np.median([ext_pos[k][1]-sy*gt_pos[k][1] for k in anchors]))
        res=float(np.median([((MM_TO_PT*gt_pos[k][0]+tx-ext_pos[k][0])**2+(sy*gt_pos[k][1]+ty-ext_pos[k][1])**2)**0.5 for k in anchors]))
        if best is None or res<best[3]: best=(sy,tx,ty,res)
    sy,tx,ty,_=best
    return (lambda x,y:(MM_TO_PT*x+tx, sy*y+ty))

def _membership(b):
    ext_cn=defaultdict(set)
    for net in b.nets.values():
        for ref,c in b.components.items():
            for p in c.pins:
                if p.connected_net==net.net_id: ext_cn[ref].add(net.net_id)
    return ext_cn

def oracle_score_page(page, gt_symbols):
    b=BipartiteGraphBuilder(cluster_eps=None)
    symbol_segs,wire_segs=SpatialClusterer.separate_wires(page.segments)
    refs,_v,net_labels=b.text_associator.associate(page)
    scale=b._estimate_scale(wire_segs) if wire_segs else b._estimate_scale(symbol_segs)
    b._build_nets(wire_segs,scale,junctions=page.junction_candidates())
    label_tol=b._derive_wire_tol(b._all_wire_segs())
    b._merge_nets_by_label(net_labels,tol=b.label_tol_factor*label_tol)
    ext_pos={r.text:r.symbol_center for r in refs}
    gt_pos={s.ref:(s.x,s.y) for s in gt_symbols}
    anchors=[k for k in ext_pos if k in gt_pos]
    if len(anchors)<2: return None
    to_pdf=_fit_translation(gt_pos,ext_pos,anchors)
    wire_tol=b._derive_wire_tol(b._all_wire_segs()); pin_tol=max(3.0*wire_tol, scale*b.pin_tol_factor)
    for sym in gt_symbols:
        comp=ComponentNode(node_id=sym.ref,ref=sym.ref,class_name="oracle")
        for j,pin in enumerate(sym.pins):
            px,py=to_pdf(pin.x,pin.y); nid=b._find_nearest_net(px,py,pin_tol)
            comp.pins.append(PinNode(pin_id=f"{sym.ref}_{j+1}",position=(px,py),connected_net=nid))
        b.components[sym.ref]=comp
    return _membership(b)

def oracle_pure_gt(sch_obj):
    gt=build_gt_graph(sch_obj); b=BipartiteGraphBuilder(cluster_eps=None)
    wire_segs=[PDFSegment(start=(w.x1,w.y1),end=(w.x2,w.y2)) for w in sch_obj.wires]
    if not wire_segs: return {"f1":0.0}
    scale=b._estimate_scale(wire_segs); b._build_nets(wire_segs,scale,junctions=[])
    wire_tol=b._derive_wire_tol(b._all_wire_segs()); pin_tol=max(3.0*wire_tol, scale*b.pin_tol_factor)
    for sym in sch_obj.symbols:
        comp=ComponentNode(node_id=sym.ref,ref=sym.ref,class_name="oracle")
        for j,pin in enumerate(sym.pins):
            nid=b._find_nearest_net(pin.x,pin.y,pin_tol)
            comp.pins.append(PinNode(pin_id=f"{sym.ref}_{j+1}",position=(pin.x,pin.y),connected_net=nid))
        b.components[sym.ref]=comp
    ext_cn=_membership(b); gt_cn=defaultdict(set)
    for nid,pins in gt.nets.items():
        for ref,_ in pins: gt_cn[ref].add(nid)
    return _greedy_f1(ext_cn,gt_cn,set(ext_cn)&set(gt.components))

def oracle_board(pdf,sch):
    sch_obj=parse_kicad_sch(sch); gt=build_gt_graph(sch_obj)
    pages=VectorExtractor().extract(str(pdf))
    ext_cn=defaultdict(set)
    for pi,page in enumerate(pages):
        pc=oracle_score_page(page,sch_obj.symbols)
        if pc is None: continue
        for ref,nets in pc.items():
            for n in nets: ext_cn[ref].add(f"{pi}:{n}")
    gt_cn=defaultdict(set)
    for nid,pins in gt.nets.items():
        for ref,_ in pins: gt_cn[ref].add(nid)
    common=set(ext_cn)&set(gt.components)
    res=_greedy_f1(ext_cn,gt_cn,common)
    res.update({"num_gt":len(gt.components),"overlap_refs":len(common),
        "num_pages":len(pages),"pure_gt_f1":oracle_pure_gt(sch_obj)["f1"]})
    return res

def main():
    names=sys.argv[1:] or [d.name for d in sorted(BASE.iterdir()) if d.is_dir() and list(d.glob("*.pdf")) and list(d.glob("*.kicad_sch"))]
    out={}
    for name in names:
        d=BASE/name; pdfs=list(d.glob("*.pdf")); schs=list(d.glob("*.kicad_sch"))
        if not (pdfs and schs): continue
        try: out[name]=oracle_board(pdfs[0],schs[0])
        except Exception as e: out[name]={"error":str(e)}
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
