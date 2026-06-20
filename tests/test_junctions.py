from src.core.graph_builder import BipartiteGraphBuilder
from src.core.pdf_parser import PDFSegment, PDFShape


def test_junction_forces_t_connection():
    # Two wires forming a cross (X intersection) but not touching endpoints to each other's segments
    # Wire 1: horizontal
    w1 = PDFSegment(start=(10.0, 20.0), end=(30.0, 20.0), item_type="line")
    # Wire 2: vertical, crossing but not terminating on w1
    w2 = PDFSegment(start=(20.0, 10.0), end=(20.0, 30.0), item_type="line")

    # Junction at the intersection
    j = PDFShape(item_type="circle", vertices=[(19.0, 19.0), (21.0, 21.0)])

    builder = BipartiteGraphBuilder(cluster_eps=5.0)
    builder._build_nets([w1, w2], scale=1.0, junctions=[j])

    assert len(builder.nets) == 1
    net = list(builder.nets.values())[0]
    # 2 real segments + 1 junction segment
    assert len(net.segments) == 3

def test_junction_no_false_positive():
    w1 = PDFSegment(start=(10.0, 20.0), end=(30.0, 20.0), item_type="line")
    w1_adj = PDFSegment(start=(30.0, 20.0), end=(40.0, 20.0), item_type="line") # forces wire_tol=0
    w2 = PDFSegment(start=(50.0, 10.0), end=(50.0, 30.0), item_type="line")
    w2_adj = PDFSegment(start=(50.0, 30.0), end=(50.0, 40.0), item_type="line")

    # Junction far from both
    j = PDFShape(item_type="circle", vertices=[(80.0, 80.0), (82.0, 82.0)])

    builder = BipartiteGraphBuilder(cluster_eps=5.0)
    builder._build_nets([w1, w1_adj, w2, w2_adj], scale=1.0, junctions=[j])

    # w1+w1_adj (net 1), w2+w2_adj (net 2), j (net 3)
    assert len(builder.nets) == 3
