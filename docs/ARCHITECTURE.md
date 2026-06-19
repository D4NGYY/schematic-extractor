# ARCHITECTURE.md — Schematic AI Reasoner

## Visione Architetturale

Il progetto è un **bridge unidirezionale**:

```
PDF Schematic → Vettore → Grafo → Netlist → LLM Reasoning
```

Nessuna ambizione di OCR full-page. Il valore è nell'estrazione geometrica precisa e nel ragionamento strutturato sul grafo risultante.

## Design Principles

1. **Estrazione prima di LLM**: Se l'estrazione fallisce, l'LLM allucina. Il 70% del valore è nell'estrazione geometrica.
2. **Ground Truth Algoritmica**: Il dataset di training non è etichettato manualmente, ma generato automaticamente mappando simboli KiCad ai cluster PDF tramite coordinate.
3. **Topologia > Semantica**: Per l'MVP, l'LLM ragiona solo sulla topologia (chi è collegato a chi), non sui modelli SPICE.
4. **Grafo Bipartito**: Componenti ↔ Nets. Questa rappresentazione è più robusta di un grafo puro dei nodi elettrici.

## Sistema di Coordinate

| Sistema | Unità | Note |
|---------|-------|------|
| KiCad `.kicad_sch` | mm | Coordinate native del formato testo |
| KiCad interno | mils (1/1000 inch) | 1 mil = 0.0254 mm |
| PDF | points (1/72 inch) | 1 pt ≈ 0.352777 mm |
| Rendering UI | px @ 300 DPI | 1 mm ≈ 11.811 px |

Tutte le conversioni sono centralizzate in `src/core/coordinate_system.py`.

## Pipeline dei Dati

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   PDF Legacy    │────▶│  PDF Parser      │────▶│  Segmenti /     │
│   (vettore)     │     │  (PyMuPDF)       │     │  Testo / Forme │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   KiCad native  │────▶│  Net Reconstructor│────▶│  Ground Truth   │
│   (.kicad_sch)  │     │  (KiCad)          │     │  (coordinate)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Schemi Synth  │────▶│  Stress Test      │────▶│  Validazione    │
│   (schemdraw)   │     │  (topologia)      │     │  (equivalenza)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                           ┌────────────────────────┐
                                           │  Clustering spaziale   │
                                           │  (DBSCAN)              │
                                           └────────────────────────┘
                                                          │
                                                          ▼
                                           ┌────────────────────────┐
                                           │  Classificazione ML    │
                                           │  (Random Forest 13D)   │
                                           └────────────────────────┘
                                                          │
                                                          ▼
                                           ┌────────────────────────┐
                                           │  Grafo Bipartito       │
                                           │  (NetworkX)              │
                                           │  Componenti ↔ Nets       │
                                           └────────────────────────┘
                                                          │
                                                          ▼
                                           ┌────────────────────────┐
                                           │  LLM Tool Calling       │
                                           │  get_neighbors, get_path│
                                           └────────────────────────┘
```

## Moduli Core

### `src/core/coordinate_system.py`
- `Vec2`: vettore 2D immutabile con operazioni base
- `CoordinateSystem`: conversioni mm ⟷ mils ⟷ points

### `src/core/pdf_parser.py` (Fase 2)
- `PDFFormatClassifier`: euristiche per formato (KiCad, Altium, Eagle, generico)
- `VectorExtractor`: estrae segmenti, archi, testo da PyMuPDF
- `TextAssociator`: associa Ref/Value al bounding box del simbolo più vicino

### `src/core/net_reconstructor.py` (Fase 0-2)
- `WireSegment`: segmento con merge collineare
- `Junction`: punto di connessione (esplicito/implicito)
- `NetLabel`: etichetta locale o globale
- `KiCadNetReconstructor`: parser .kicad_sch, wire_merge, junction_detect, build_nets

### `src/core/graph_builder.py` (Fase 3)
- Costruisce grafo bipartito Componenti ↔ Nets
- Pin-point matching con stub virtuali (3-5px, confidence 0.7)
- Export SPICE `.cir` e KiCad `.net`

### `src/ml/classifier.py` (Fase 3)
- Feature vector 13 dimensioni per ogni cluster
- Random Forest primario, CNN leggera fallback
- Target: >90% accuracy sulle 10 classi base

### `src/ui/app.py` (Fase 6)
- Streamlit con pre-render PNG 300 DPI
- Overlay SVG absolute-positioned per highlight
- `st.selectbox` per selezione nets/componenti (no click nativo)

## Trade-off Architetturali

### Grafo Bipartito vs Grafo dei Nodi

Abbiamo scelto il **grafo bipartito** (Componenti ↔ Nets) perché:
- Preserva l'informazione sui componenti (non solo i nodi elettrici)
- Permette query tipo "quali componenti sono collegati a R1?"
- È più robusto alle ambiguità di net naming

Svantaggio: rappresentazione meno compatta. Compensato con caching NetworkX.

### Random Forest vs CNN

Random Forest è il classificatore primario perché:
- Feature interpretabili (aspect ratio, solidity, ecc.)
- Training veloce con dataset piccoli
- Non richiede GPU

CNN leggera è fallback per:
- Simboli con aspect ratio ambiguo (resistor vs inductor)
- Orientazioni non ortogonali (R&D, post-MVP)

## Validazione

### Gate Quantitativi
- **LOO-CV**: >98% precision pin-to-net, >90% recall componenti
- **Net Equivalence**: due netlist sono equivalenti se l'insieme dei pin collegati per componente è identico, a meno del renaming dei net IDs

### ERC Topologico
- Pin flottanti (nessun net)
- Componenti isolati (0 nets)
- Nets dangling
- Nets globali identiche ma disconnesse topologicamente (Warning)

## Logging

Ogni componente estratto logga:
- `phase`: parsing, clustering, classification, graph_building
- `confidence`: 0.0-1.0
- `heuristic`: quale euristica ha prodotto il risultato

Usa `structlog` con formato JSON per tracciabilità.

## Estensioni Future (R&D)

- **OCR su ref descrizioni**: solo se il classificatore è insicuro
- **Simboli non standard**: clustering density-based con fallback manuale
- **Differential pairs**: riconoscimento coppie di linee parallele con spaziamento costante
- **Bus expansion**: sintassi `NAME[M..N]` con bus entry

## Sicurezza e Limitazioni

- Nessun model execution SPICE (solo topologia)
- Nessun upload di file verso LLM provider (i dati restano locali con Ollama)
- Rate-limiting e tracking costi per provider cloud
