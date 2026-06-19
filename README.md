# Schematic AI Reasoner

> **Bridge tra file PDF legacy (non editabili) e l'ecosistema moderno** di simulazione e versioning.

## Visione

Trasformare schemi elettrici in PDF (o immagini) in un grafo netlist navigabile, ragionabile tramite LLM, ed esportabile in SPICE/KiCad.

## Pipeline

1. **Estrazione Vettoriale** (PyMuPDF) → segmenti, testo, forme
2. **Clustering ML** (Random Forest) → classificazione componenti
3. **Reconstruction Netlist** (Grafo Bipartito) → pins, nets, junctions
4. **LLM Reasoning** (Tool Calling) → analisi topologica, Q&A
5. **UI Streamlit** → visualizzazione, export, validazione

## Struttura

```
schematic_extractor/
├── scripts/              # Utility one-off (dataset, export)
├── src/
│   ├── core/             # Estrazione, grafo, netlist
│   ├── ml/               # Classificazione ML
│   └── ui/               # Streamlit app
├── tests/                # Unit test con mock geometrici
├── data/
│   ├── kicad/            # Schemi KiCad sorgente
│   ├── pdf/              # PDF esportati
│   └── ground_truth/     # Netlist di riferimento
└── docs/                 # ARCHITECTURE.md, dataset notes
```

## Setup

```bash
pip install -e ".[dev]"
```

## Fase Corrente

**Fase 0: Foundation e Dataset Elettrico**
- [ ] Selezionare 5 schemi KiCad semplici
- [ ] `kicad_net_reconstructor.py`: wire_merge, junction_detect, label_global_scope
- [ ] Esportare PDF standardizzati
- [ ] `align_kicad_to_pdf.py`: mappatura automatica simboli → cluster
- [ ] `validate_net_equivalence()`: check equivalenza netlist

## Requisiti di Sistema

- Python 3.12+
- KiCad (per export PDF e generazione dataset)

## Licenza

MIT
