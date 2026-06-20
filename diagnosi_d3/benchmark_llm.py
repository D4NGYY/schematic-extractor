"""In-process LLM tool-calling benchmark with deterministic scoring.

Builds the Bryston graph once, then runs each candidate model over the 5
benchmark queries via SchematicAgent, recording the tool-call trace, the final
answer, latency, and a 0-5 score against hand-checked ground truth.

Run: uv run python diagnosi_d3/benchmark_llm.py
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.pdf_parser import VectorExtractor
from src.core.text_associator import TextAssociator
from src.llm.agent import OllamaClient, SchematicAgent
from src.llm.tools import GraphContext

MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "mistral:7b-instruct-v0.3-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
]

PDF = "test_input/bryston_schematic.pdf"
QUERY_TIMEOUT_S = 150

# Each query: text + expected tool + key arg substrings + ground-truth entities.
QUERIES = [
    {
        "id": "Q1",
        "text": "Quali componenti sono isolati?",
        "tool": "find_isolated",
        "arg_keys": [],
        "entities": ["RB14", "R45", "DX7"],
    },
    {
        "id": "Q2",
        "text": "Elenca i componenti collegati a WB1",
        "tool": "get_neighbors",
        "arg_keys": ["WB1"],
        "entities": ["PR2", "RB10"],
    },
    {
        "id": "Q3",
        "text": "Trova il path che collega RF1 a R37",
        "tool": "get_path",
        "arg_keys": ["RF1", "R37"],
        "entities": ["RF1", "R37"],
    },
    {
        "id": "Q4",
        "text": "Cerca i componenti con valore 10k",
        "tool": "search_by_value",
        "arg_keys": ["10k"],
        "entities": ["WB1", "D1"],
    },
    {
        "id": "Q5",
        "text": "Quali net collegano almeno 2 componenti?",
        "tool": "get_nets_summary",
        "arg_keys": [],
        "entities": ["Net-6", "Net-46", "Net-81"],
    },
]


class RecordingAgent(SchematicAgent):
    """Agent that records every (tool_name, args) dispatched during a query."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trace: list[tuple[str, str]] = []
        self.iterations_used = 0

    def _execute_tool(self, name: str, arguments_str: str) -> str:
        self.trace.append((name, arguments_str))
        return super()._execute_tool(name, arguments_str)


def score_query(q: dict[str, Any], trace: list[tuple[str, str]], answer: str,
                maxed_out: bool) -> tuple[int, dict[str, bool]]:
    """0-5 tool-gated rubric. Args/data credit REQUIRE the correct tool to have
    actually run, so entities echoed from the question text earn nothing:
      correct_tool (+2) | correct_args (+1) | data_present (+1) | clean (+1)."""
    tools_called = [name for name, _ in trace]
    args_blob = " ".join(args for _, args in trace).lower()
    ans_lower = answer.lower()

    correct_tool = q["tool"] in tools_called
    args_ok = correct_tool and all(k.lower() in args_blob for k in q["arg_keys"])
    data_ok = correct_tool and all(e.lower() in ans_lower for e in q["entities"])
    no_loop = not maxed_out and len(trace) <= len(QUERIES) + 4
    clean = no_loop and bool(answer.strip()) and "could not answer" not in ans_lower

    checks = {
        "correct_tool": correct_tool,
        "correct_args": args_ok,
        "data_present": data_ok,
        "clean_answer": clean,
    }
    return (2 if correct_tool else 0) + sum(checks[k] for k in ("correct_args", "data_present", "clean_answer")), checks


async def run_query(graph_context: GraphContext, model: str, q: dict[str, Any]) -> dict[str, Any]:
    client = OllamaClient(model=model)
    agent = RecordingAgent(graph_context=graph_context, llm_client=client, max_iterations=10)
    start = time.perf_counter()
    try:
        answer = await asyncio.wait_for(agent.query(q["text"]), timeout=QUERY_TIMEOUT_S)
        timed_out = False
    except TimeoutError:
        answer = "[TIMEOUT]"
        timed_out = True
    elapsed = time.perf_counter() - start

    maxed_out = "could not answer within max iterations" in answer.lower() or timed_out
    score, checks = score_query(q, agent.trace, answer, maxed_out)
    return {
        "query_id": q["id"],
        "query": q["text"],
        "tool_trace": agent.trace,
        "answer": answer.strip(),
        "elapsed_s": round(elapsed, 2),
        "timed_out": timed_out,
        "score": score,
        "checks": checks,
    }


async def main() -> None:
    print(f"Extracting graph from {PDF} ...")
    pages = VectorExtractor().extract(PDF)
    graph = BipartiteGraphBuilder(text_associator=TextAssociator()).build_from_page(pages[0])
    graph_context = GraphContext(graph)
    print(f"Graph: {len(graph_context.components)} components, {len(graph_context.nets)} nets\n")

    report: dict[str, Any] = {"pdf": PDF, "models": {}}
    for model in MODELS:
        print(f"==== {model} ====")
        results = []
        for q in QUERIES:
            res = await run_query(graph_context, model, q)
            results.append(res)
            mark = "PASS" if res["score"] >= 3 else "FAIL"
            tools = ",".join(t for t, _ in res["tool_trace"]) or "-"
            print(f"  {res['query_id']} [{mark} {res['score']}/5] {res['elapsed_s']}s "
                  f"tools=[{tools}]")
        total = sum(r["score"] for r in results)
        passes = sum(1 for r in results if r["score"] >= 3)
        avg_time = round(sum(r["elapsed_s"] for r in results) / len(results), 2)
        report["models"][model] = {
            "results": results,
            "total_score": total,
            "queries_passed": passes,
            "avg_time_s": avg_time,
        }
        print(f"  TOTAL {total}/25  passed {passes}/5  avg {avg_time}s\n")

    # Winner: most queries passed, then highest total score, then fastest.
    ranking = sorted(
        report["models"].items(),
        key=lambda kv: (kv[1]["queries_passed"], kv[1]["total_score"], -kv[1]["avg_time_s"]),
        reverse=True,
    )
    report["winner"] = ranking[0][0]
    print("WINNER:", report["winner"])

    out = Path("diagnosi_d3/benchmark_llm_results.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    asyncio.run(main())
