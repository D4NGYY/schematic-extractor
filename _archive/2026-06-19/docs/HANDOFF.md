# HANDOFF — Schematic AI Reasoner

> **Sessione:** 2024-06-18  
> **Stato:** Fasi 0-3 completate e funzionanti. Fasi 4-6 da fare.  
> **Branch:** `main` (nessun branch, tutto su working directory)  

---

## 1. Cosa è stato costruito

### Fase 0 — Foundation e Dataset Elettrico ✅

| File | Scopo | Stato |
|------|-------|-------|
| `data/kicad/test_resistor_divider.kicad_sch` | Schema KiCad reale (partitore resistivo) | ✅ Funziona |
| `data/kicad/synthetic/*.kicad_sch` | 7 schemi sintetici (stress test) | ✅ Generati |
| `scripts/kicad_net_reconstructor.py` | Parser `.kicad_sch`, wire_merge, junction_detect, build_nets | ✅ 17 test |
| `scripts/validate_net_equivalence.py` | Check equivalenza topologica netlist | ✅ Funziona |
| `scripts/align_kicad_to_pdf.py` | Mappatura simboli KiCad → cluster PDF | ✅ Stub |
| `scripts/generate_synthetic_schematics.py` | Generatore schemi sintetici | ✅ 7 schemi, tutti PASS |

**Test:** `pytest tests/test_kicad_net_reconstructor.py` → 17/17 passano.

**Stress test:** `python scripts/generate_synthetic_schematics.py` → tutti i 7 schemi convergono a netlist valide.

### Fase 1 — Architettura e Tooling ✅

| File | Scopo | Stato |
|------|-------|-------|
| `pyproject.toml` | Build, dipendenze, ruff, mypy, pytest | ✅ |
| `src/core/coordinate_system.py` | Vec2 + mm/mils/points | ✅ 2 test |
| `src/core/logging_config.py` | structlog JSON/console | ✅ |
| `src/core/__init__.py` | | ✅ |
| `src/ml/__init__.py` | | ✅ |
| `src/ui/__init__.py` | | ✅ |
| `src/__init__.py` | | ✅ |

**Linting:** `ruff check .` → ✅ passa.
**Type checking:** `mypy .` → 10 errori minori (fix in corso, vedi sezione 3).

### Fase 2 — Parser PDF Vettoriale ✅

| File | Scopo | Stato |
|------|-------|-------|
| `src/core/pdf_parser.py` | `VectorExtractor`, `PDFFormatClassifier`, `PDFSegment`, `PDFShape`, `PDFTextBlock` | ✅ 10 test |
| `src/core/text_associator.py` | Associazione Ref/Value/NetLabel con priorità direzionale | ✅ 4 test |

**Test:** `pytest tests/test_pdf_parser.py` → 10/10 passano.

### Fase 3 — Clustering, ML, Grafo Bipartito ✅

| File | Scopo | Stato |
|------|-------|-------|
| `src/ml/clustering.py` | DBSCAN spaziale adattivo | ✅ 3 test |
| `src/ml/feature_extractor.py` | Feature vector 13D (aspect ratio, solidity, convex hull...) | ✅ 2 test |
| `src/ml/classifier.py` | Random Forest, 10 classi, fallback <0.7 | ✅ 2 test |
| `src/core/graph_builder.py` | Grafo bipartito Componenti ↔ Nets, export SPICE/KiCad/JSON | ✅ 6 test |

**Test:** `pytest tests/test_graph_builder.py` → 6/6 passano.

**Totale test:** 44/44 passano.

---

## 2. Struttura del progetto

```
schematic_extractor/
├── scripts/                              # Utility one-off
│   ├── kicad_net_reconstructor.py       # Core Fase 0
│   ├── validate_net_equivalence.py        # Check equivalenza
│   ├── generate_synthetic_schematics.py  # Stress test suite
│   └── align_kicad_to_pdf.py            # Training set auto
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── coordinate_system.py         # Vec2 + conversioni
│   │   ├── logging_config.py            # structlog
│   │   ├── pdf_parser.py                # PyMuPDF extraction
│   │   ├── text_associator.py           # Ref/Value/NetLabel assoc
│   │   └── graph_builder.py             # Grafo bipartito + export
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── clustering.py                # DBSCAN
│   │   ├── feature_extractor.py          # Feature vector 13D
│   │   └── classifier.py                # Random Forest
│   └── ui/
│       └── __init__.py                   # (vuoto, Fase 6)
├── tests/
│   ├── test_kicad_net_reconstructor.py   # 17 test
│   ├── test_pdf_parser.py               # 10 test
│   └── test_graph_builder.py             # 6 test
├── data/
│   ├── kicad/
│   │   ├── test_resistor_divider.kicad_sch
│   │   └── synthetic/                    # 7 schemi generati
│   └── ground_truth/                     # (output test)
├── docs/
│   ├── ARCHITECTURE.md                   # Pipeline, trade-off, design
│   └── HANDOFF.md                        # Questo file
├── pyproject.toml
├── README.md
└── .venv/                                # Ambiente Python (uv)
```

---

## 3. Problemi aperti

### 3.1 Mypy — 10 errori (non bloccanti, ruff passa)

```
src/ml/feature_extractor.py:93-96
  → np.mean/np.std ritorna floating[Any] | float, FeatureVector attesa float
  → Fix: cast esplicito float(centroid_x), float(std_x), ...

src/core/graph_builder.py:267-275
  → Funzione cross() interna senza type annotation
  → Fix: aggiungere tipi ai parametri, annotare return
```

**Comando per riprodurre:**
```bash
uv run mypy .
```

### 3.2 Modello ML non addestrato

Il `ComponentClassifier` ha un metodo `fit()` ma nessuno script lo chiama. Il grafo bipartito gestisce questo caso fallback (classifica come "unknown" con confidence 0.0).

**Per addestrare:**
```python
from src.ml.classifier import ComponentClassifier
import numpy as np
clf = ComponentClassifier()
X = np.array([...])  # feature vectors 13D
y = np.array([...])  # label
clf.fit(X, y)
clf.save("models/classifier.pkl")
```

**Per usare in produzione:**
```python
clf = ComponentClassifier("models/classifier.pkl")
```

### 3.3 KiCad CLI non disponibile

L'export PDF da KiCad (`kicad-cli sch export pdf`) richiede KiCad installato. Gli schemi sintetici sono in `.kicad_sch` ma non hanno un PDF corrispondente.

**Workaround:** usare `schemdraw` o esportare manualmente da KiCad GUI.

---

## 4. Come riprendere

### 4.1 Setup ambiente (se nuova macchina)

```bash
cd C:\Users\danie\Desktop\Projects\schematic_extractor
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
```

### 4.2 Verifica stato

```bash
.venv\Scripts\python.exe -m pytest tests/ -q       # 44 test
.venv\Scripts\ruff.exe check .                        # 0 errori
.venv\Scripts\mypy.exe .                             # 10 errori (fixare)
```

### 4.3 Pipeline end-to-end da testare

```python
from src.core.pdf_parser import VectorExtractor
from src.core.graph_builder import BipartiteGraphBuilder
from src.ml.classifier import ComponentClassifier

# 1. Estrai PDF
pages = VectorExtractor().extract("path/to/schematic.pdf")

# 2. Costruisci grafo (senza modello addestrato → classifica "unknown")
builder = BipartiteGraphBuilder()
graph = builder.build_from_page(pages[0])

# 3. Esporta
builder.export_json("output.json")
builder.export_spice("output.cir")
builder.export_kicad_netlist("output.net")
```

### 4.4 KiCad → Netlist (ground truth)

```bash
python scripts/kicad_net_reconstructor.py reconstruct \
  data/kicad/test_resistor_divider.kicad_sch \
  --output output.json
```

---

## 5. Prossimi task (Fasi 4-6)

| # | Fase | Task | File previsto | Complessità |
|---|------|------|-------------|-------------|
| 1 | 3 | Fix 10 errori mypy | `feature_extractor.py`, `graph_builder.py` | 🟢 Bassa |
| 2 | 4 | ERC topologico (pin flottanti, isolati, dangling) | `src/core/erc.py` | 🟡 Media |
| 3 | 4 | LOO-CV + metriche | `tests/test_erc.py` | 🟡 Media |
| 4 | 5 | LLM Tool Calling (get_neighbors, get_path, get_net_components) | `src/core/llm_tools.py` | 🔴 Alta |
| 5 | 5 | Benchmark topologico | `tests/benchmark_topology.py` | 🟡 Media |
| 6 | 6 | Streamlit UI (pre-render PNG, overlay SVG, selectbox) | `src/ui/app.py` | 🟡 Media |
| 7 | 6 | Portfolio 3 casi studio | `docs/portfolio.md` | 🟢 Bassa |

---

## 6. Decisioni architetturali prese

- **Grafo bipartito** (Componenti ↔ Nets) invece di grafo puro nodi
- **Random Forest** primario, fallback "unknown" sotto confidence 0.7
- **Feature vector 13D** con log1p normalizzazione per stabilità
- **No OCR** su ref designator (solo geometria + euristiche)
- **structlog** JSON in produzione, console colorato in dev
- **ruff** per linting+formatting, **mypy** strict per type checking

---

## 7. Dipendenze chiave

| Pacchetto | Versione | Uso |
|-----------|----------|-----|
| PyMuPDF | >=1.23.0 | Parser PDF vettoriale |
| scikit-learn | >=1.3.0 | DBSCAN, Random Forest |
| scipy | >=1.11.0 | ConvexHull, pdist |
| networkx | >=3.0 | Grafo bipartito |
| numpy | >=1.26.0 | Array ops |
| structlog | >=23.0.0 | Logging |
| pydantic | >=2.0.0 | Validazione (previsto Fase 6) |
| typer | >=0.9.0 | CLI |
| pytest | >=7.4.0 | Test |
| ruff | >=0.1.0 | Linting |
| mypy | >=1.6.0 | Type checking |
| streamlit | >=1.28.0 | UI (Fase 6) |

---

*Documento generato automaticamente. Ultimo aggiornamento: sessione corrente.*
