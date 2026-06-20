# TEST_MANUAL — Phase 5 LLM tool calling (Ollama, end-to-end)

_Last run: 2026-06-20. Branch: `feat/wire-symbol-separation`._

This document is the manual/benchmark report for the LLM tool-calling layer: how
to reproduce the benchmark, the ground truth, the bugs the real-Ollama run
exposed, and the prompt-engineering iterations.

---

## 1. Prerequisites

```bash
ollama serve                                   # API on http://localhost:11434
ollama pull qwen2.5:7b-instruct-q4_K_M         # winner (~4.7 GB)
ollama pull mistral:7b-instruct-v0.3-q4_K_M    # benchmark candidate
ollama pull llama3.1:8b-instruct-q4_K_M        # benchmark candidate / baseline
```

Hardware target: 3070 Ti 8 GB VRAM → Q4_K_M 7-8B models.

---

## 2. Run

```bash
# Full scored benchmark (3 models x 5 queries, in-process, deterministic scoring)
uv run python diagnosi_d3/benchmark_llm.py          # -> diagnosi_d3/benchmark_llm_results.json

# Single real query via CLI (default model = qwen2.5)
uv run schematic-extractor query "Quali componenti sono isolati?" \
    --pdf test_input/bryston_schematic.pdf

# Mock mode (no Ollama, for plumbing tests only)
uv run schematic-extractor query "..." --pdf test_input/bryston_schematic.pdf --mock

# Streamlit chat (model selector defaults to the benchmark winner)
streamlit run src/ui/app.py
```

---

## 3. Benchmark queries + ground truth (Bryston page 0)

Graph: **13 components, 125 nets**. Queries adapted to real refs (the schematic
has no `R1`/`U1`; refs are `WB1, R37, RB2, ...`).

| # | Query | Expected tool | Ground-truth answer |
|---|-------|---------------|---------------------|
| Q1 | Quali componenti sono isolati? | `find_isolated` | RB14, R45, DX7 (3) |
| Q2 | Elenca i componenti collegati a WB1 | `get_neighbors` | PR2, RB10 |
| Q3 | Trova il path che collega RF1 a R37 | `get_path` | RF1 → Net-81 → R37 |
| Q4 | Cerca i componenti con valore 10k | `search_by_value` | WB1, D1 (10K0) |
| Q5 | Quali net collegano almeno 2 componenti? | `get_nets_summary` | Net-6, Net-46, Net-81 |

Scoring (0-5, tool-gated): `correct_tool (+2)` · `correct_args (+1, requires
tool)` · `data_present (+1, requires tool — entities echoed from the question
earn nothing)` · `clean/no-loop (+1)`. Pass = score ≥ 3.

---

## 4. Results

| Model | Total | Queries passed | Avg time |
|-------|-------|----------------|----------|
| **qwen2.5:7b-instruct-q4_K_M** 🏆 | **25/25** | **5/5** | **3.24 s** |
| llama3.1:8b-instruct-q4_K_M | 21/25 | 4/5 | 3.67 s |
| mistral:7b-instruct-v0.3-q4_K_M | 20/25 | 4/5 | 4.64 s |

**Winner: qwen2.5:7b** — perfect score and fastest. Set as `DEFAULT_MODEL` in
`src/llm/agent.py` (consumed by `OllamaClient`, the Typer CLI and the Streamlit
selector).

Discriminating failure: on Q2 both mistral and llama call `get_net_components`
(WB1 is a *component*, not a net) instead of `get_neighbors` → wrong-tool, 1/5.
qwen disambiguates the two tools correctly.

---

## 5. Bugs the real run exposed (the `--mock` benchmark had hidden them)

The pre-existing handoff claimed "Qwen 5/5". That benchmark ran against a broken
`GraphContext` returning empty data — the models hallucinated plausible answers.
Root causes found and fixed (all in `src/llm/`):

1. **Schema mismatch (critical).** `GraphContext` classified nodes by
   `data["type"]` and read pins from `edge_data["pin"]`, but
   `BipartiteGraphBuilder` emits `bipartite=0/1` and `pin_id`. On any real graph
   `self.components`/`self.nets` were empty → every tool returned "not found".
   Tests passed only because the fixtures used the old `type=`/`pin=` schema.
   Fix: `_is_component`/`_is_net`/`_net_name` helpers accept both schemas; pins
   read `pin_id` then `pin`.
2. **`_execute_tool` crash.** `method(**kwargs)` raised when a model passed a
   non-object JSON (mistral sent a bare string). Fix: reject non-dict args with a
   clear error the model can recover from.
3. **`get_nets_summary` crash.** `len(comps) >= min_components` raised when a
   model passed `"2"` (string). Fix: coerce `min_components` to int.

## 6. Prompt-engineering / parser iterations

- **v1 (inherited):** Llama-centric system prompt + ReAct parser matching only
  `TOOL_CALL: name({...})`. Local models emit tool calls in other text shapes,
  all missed → false "answers".
- **v2 (this session):** model-agnostic prompt (prefer native function calling,
  explicit single-line fallback, "never invent, report empty/errors honestly").
  ReAct parser hardened, gated on real tool names, to also accept:
  - envelope `{"name": "tool", "arguments": {...}}` (qwen),
  - prefixed/bare object `brtc_get_path {...}` (qwen),
  - positional `get_path("RF1", "R37")` mapped to schema param order (mistral).

Regression tests: `tests/test_llm_tools.py` (real `bipartite` schema +
`get_nets_summary`), `tests/test_llm_agent.py` (3 parser shapes, non-dict guard,
no-false-positive on prose). Full suite: **192 passed**.
