from __future__ import annotations

from src.core.pdf_parser import PDFSegment, PDFShape, PDFTextBlock
from src.ml.clustering import ComponentCluster, SpatialClusterer

def test_text_guided_merge_merges_nearby_fragments() -> None:
    # Creiamo due cluster separati che altrimenti non verrebbero uniti
    seg1 = PDFSegment(start=(10, 10), end=(20, 10))
    seg2 = PDFSegment(start=(30, 10), end=(40, 10))
    
    # E un testo "U1" che sta in mezzo o li copre
    text = PDFTextBlock(text="U1", bbox=(15, 5, 35, 15))
    
    clusterer = SpatialClusterer(eps=2.0, min_samples=1)
    # 10 e 30 sono distanti 10 (gap da 20 a 30), quindi eps=2 non li unisce
    clusters = clusterer.cluster([seg1, seg2], [], [text])
    
    # Con il text guided merge, i centri dei cluster (15,10 e 35,10)
    # L'area di gravità di "U1" si espande:
    # bbox U1: 15, 5, 35, 15 (w=20, h=10)
    # pad_x = max(15, 30) = 30
    # pad_y = max(15, 15) = 15
    # gz = (15-30, 5-15, 35+30, 15+15) = (-15, -10, 65, 30)
    # Entrambi i centri cadono nella GZ, quindi dovrebbero fondersi in 1 cluster.
    assert len(clusters) == 1
    assert clusters[0].num_segments == 2

def test_text_guided_merge_ignores_distant_fragments() -> None:
    seg1 = PDFSegment(start=(10, 10), end=(20, 10))
    seg2 = PDFSegment(start=(100, 100), end=(110, 100))
    
    # "U1" vicino a seg1
    text = PDFTextBlock(text="U1", bbox=(15, 5, 25, 15))
    
    clusterer = SpatialClusterer(eps=2.0, min_samples=1)
    clusters = clusterer.cluster([seg1, seg2], [], [text])
    
    # Il seg2 è troppo distante dalla GZ di U1, quindi avremo 2 cluster
    assert len(clusters) == 2

def test_text_guided_merge_ignores_non_refs() -> None:
    seg1 = PDFSegment(start=(10, 10), end=(20, 10))
    seg2 = PDFSegment(start=(30, 10), end=(40, 10))
    
    # "Hello" non è un ref designator
    text = PDFTextBlock(text="Hello", bbox=(15, 5, 35, 15))
    
    clusterer = SpatialClusterer(eps=2.0, min_samples=1)
    clusters = clusterer.cluster([seg1, seg2], [], [text])
    
    # Non uniti perché il testo non matcha REF_PATTERN
    assert len(clusters) == 2
