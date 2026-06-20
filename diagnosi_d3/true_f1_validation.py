import json
import logging
from pathlib import Path
from collections import defaultdict

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.kicad_gt_reader import build_gt_graph, parse_kicad_sch
from src.core.pdf_parser import VectorExtractor

logger = logging.getLogger("true_f1")
logging.basicConfig(level=logging.INFO)

def run_true_f1():
    base_dir = Path("test_input/multi_schematic")
    boards = []
    
    for b in ["arduino_micro", "arduino_nano"]:
        d = base_dir / b
        pdfs = list(d.glob("*.pdf"))
        schs = list(d.glob("*.kicad_sch"))
        if pdfs and schs:
            boards.append({
                "name": b,
                "pdf": pdfs[0],
                "sch": schs[0]
            })
            
    results = {}
    
    for board in boards:
        name = board["name"]
        logger.info(f"Processing {name}")
        
        # Extractor
        extractor = VectorExtractor()
        pages = extractor.extract(str(board["pdf"]))
        if not pages:
            continue
        page = pages[0]
        
        # Graph builder
        builder = BipartiteGraphBuilder(cluster_eps=None)
        builder.build_from_page(page)
        
        # GT
        gt_sch = parse_kicad_sch(board["sch"])
        gt_graph = build_gt_graph(gt_sch)
        
        # 1. Component matching
        ext_refs = set(builder.components.keys())
        gt_refs = set(gt_graph.components)
        common_refs = ext_refs.intersection(gt_refs)
        
        # Old F1 logic (for reporting)
        ext_net_lists = []
        for net in builder.nets.values():
            net_pins = []
            for ref, c in builder.components.items():
                for p in c.pins:
                    if p.connected_net == net.net_id:
                        net_pins.append((ref, p.pin_id))
            if net_pins:
                ext_net_lists.append(net_pins)
                
        gt_net_lists = [list(pins) for pins in gt_graph.nets.values() if len(pins) > 1]
        
        def calc_old_f1(e_nets, g_nets):
            e_pairs = set()
            for net in e_nets:
                pins = sorted(net)
                for i in range(len(pins)):
                    for j in range(i + 1, len(pins)):
                        e_pairs.add((pins[i], pins[j]))
            g_pairs = set()
            for net in g_nets:
                pins = sorted(net)
                for i in range(len(pins)):
                    for j in range(i + 1, len(pins)):
                        g_pairs.add((pins[i], pins[j]))
            if not g_pairs and not e_pairs: return 1.0
            if not g_pairs or not e_pairs: return 0.0
            tp = len(e_pairs & g_pairs)
            fp = len(e_pairs - g_pairs)
            fn = len(g_pairs - e_pairs)
            p = tp / (tp + fp) if tp + fp > 0 else 0
            r = tp / (tp + fn) if tp + fn > 0 else 0
            return 2 * p * r / (p + r) if p + r > 0 else 0.0
            
        f1_old = calc_old_f1(ext_net_lists, gt_net_lists)
        
        # 2. Extract PIN-AGNOSTIC connections
        ext_comp_nets = defaultdict(set)
        for net in builder.nets.values():
            for ref, c in builder.components.items():
                for p in c.pins:
                    if p.connected_net == net.net_id:
                        ext_comp_nets[ref].add(net.net_id)
                        
        gt_comp_nets = defaultdict(set)
        for nid, pins in gt_graph.nets.items():
            for ref, _ in pins:
                gt_comp_nets[ref].add(nid)
                
        # 3. Net mapping (greedy bipartite)
        ext_net_refs = defaultdict(set)
        for ref, nets in ext_comp_nets.items():
            if ref in common_refs:
                for n in nets:
                    ext_net_refs[n].add(ref)
                    
        gt_net_refs = defaultdict(set)
        for ref, nets in gt_comp_nets.items():
            if ref in common_refs:
                for n in nets:
                    gt_net_refs[n].add(ref)
                    
        net_mapping = {}
        scores = []
        for en, e_refs in ext_net_refs.items():
            for gn, g_refs in gt_net_refs.items():
                overlap = len(e_refs & g_refs)
                if overlap > 0:
                    scores.append((overlap, en, gn))
                    
        scores.sort(reverse=True)
        used_ext = set()
        used_gt = set()
        for overlap, en, gn in scores:
            if en not in used_ext and gn not in used_gt:
                net_mapping[en] = gn
                used_ext.add(en)
                used_gt.add(gn)
                
        # 4. Calculate True Positives
        tp = 0
        fp = 0
        fn = 0
        
        for ref in common_refs:
            e_nets = ext_comp_nets[ref]
            g_nets = gt_comp_nets[ref]
            
            e_mapped_nets = set()
            for n in e_nets:
                if n in net_mapping:
                    e_mapped_nets.add(net_mapping[n])
                else:
                    fp += 1
                    
            tp += len(e_mapped_nets & g_nets)
            fp += len(e_mapped_nets - g_nets)
            fn += len(g_nets - e_mapped_nets)
            
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_new = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        
        metrics = {
            "num_ext": len(ext_refs),
            "num_gt": len(gt_refs),
            "overlap_refs": len(common_refs),
            "f1_old": f1_old,
            "f1_new": f1_new,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        
        results[name] = metrics
        
        with open(f"diagnosi_d3/true_f1_{name}.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_true_f1()
