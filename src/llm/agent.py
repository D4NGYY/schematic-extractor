import json
import os
import re
from typing import Any

import openai
import structlog

from src.llm.tools import TOOLS_SCHEMA, GraphContext

logger = structlog.get_logger("agent")

# Default tool-calling model: winner of the 5-query Bryston benchmark
# (qwen2.5 scored 25/25, 5/5 queries, avg 3.24s — see diagnosi_d3/benchmark_llm.py).
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"

# Ollama endpoint. In Docker (compose) point this at the ollama service via the
# OLLAMA_BASE_URL env var; locally it defaults to the loopback.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Derived from TOOLS_SCHEMA: valid tool names and their ordered parameter names.
# Used by the ReAct fallback parser to recognise text-form tool calls while
# staying gated on real tool names (avoids false positives in prose).
_VALID_TOOLS = {t["function"]["name"] for t in TOOLS_SCHEMA}
_TOOL_PARAMS = {
    t["function"]["name"]: list(t["function"]["parameters"].get("properties", {}).keys())
    for t in TOOLS_SCHEMA
}

class LLMClient:
    """Abstract interface for LLM clients."""
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Ollama client using OpenAI SDK compatibility layer."""
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = openai.AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"  # required by SDK but ignored by Ollama
        )

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
        # We pass tools only if provided
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message


class MockClient(LLMClient):
    """Mock client for testing without network."""
    def __init__(self) -> None:
        self.call_count = 0

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
        self.call_count += 1
        # Simple mock logic: first turn calls a tool, second turn answers.

        class MockMessage:
            def __init__(self, content: str | None, tool_calls: list[Any] | None = None) -> None:
                self.content = content
                self.tool_calls = tool_calls

        class MockToolCall:
            def __init__(self, name: str, arguments: str) -> None:
                self.id = f"call_{name}"
                class Function:
                    name: str
                    arguments: str
                self.function = Function()
                self.function.name = name
                self.function.arguments = arguments

        if self.call_count == 1:
            # First turn: trigger a tool call for get_neighbors
            return MockMessage(
                content=None,
                tool_calls=[MockToolCall("get_neighbors", '{"component_ref": "R1"}')]
            )
        else:
            # Second turn: provide the answer
            return MockMessage(content="R1 is connected to U1 via net N1.")


class SchematicAgent:
    """Agent loop handling LLM conversation and tool dispatching."""

    def __init__(self, graph_context: GraphContext, llm_client: LLMClient, max_iterations: int = 10):
        self.graph_context = graph_context
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        # Messaggi dell'ultima query (per tool-trace UI). Aggiornato da query().
        self.last_messages: list[dict[str, Any]] = []
        self.system_prompt = (
            "You are an electrical-engineering assistant answering questions about an "
            "electronic schematic, reconstructed as a Components<->Nets graph by automated "
            "extraction (not a hand-verified netlist).\n"
            "\n"
            "DOMAIN KNOWLEDGE — use this to interpret tool results:\n"
            "- Ref designators: R=resistor, C=capacitor, L=inductor, D=diode, Q=transistor, "
            "U=IC (integrated circuit), J/P=connector, Y=crystal, F=fuse, K=relay, SW=switch. "
            "Tool outputs include a human-readable 'description' field (e.g. 'R1 (resistor, 10k)') — prefer it.\n"
            "- Nets labeled GND/VCC/VDD/+5V/3V3 etc. are POWER RAILS (shared supply/ground); "
            "tool outputs tag them as 'power rail'. All other nets are SIGNAL nets.\n"
            "- 'Isolated' components (find_isolated) have no detected connection — often an "
            "extraction gap, NOT necessarily a design intent.\n"
            "\n"
            "RULES:\n"
            "1. Base EVERY claim on tool results. Never invent components, nets, values or connections.\n"
            "2. Reply in the user's language. Be concise and technical.\n"
            "3. When asked 'what is this circuit' / 'describe the schematic', do NOT just list refs. "
            "Call get_nets_summary(min_components=2) and find_isolated first, then EXPLAIN the "
            "topology in functional terms: identify the rails, group components by function "
            "(filter, amplifier, regulator, digital, ...), and state what the circuit likely does.\n"
            "4. Cite exact refs/net names from the tool output. Prefer the 'description' field over raw refs.\n"
            "5. If a tool returns an error or empty result, report that honestly instead of guessing.\n"
            "\n"
            "TOOL CALL FORMAT:\n"
            "- PREFER native function-calling. If your backend cannot, fall back to a single line:\n"
            "  TOOL_CALL: tool_name({\"arg\": \"value\"})\n"
            "- Call one tool at a time, read its JSON result, then continue or answer."
        )

    def _execute_tool(self, name: str, arguments_str: str) -> str:
        """Executes a tool on GraphContext and returns the JSON result string."""
        try:
            kwargs = json.loads(arguments_str) if arguments_str.strip() else {}
        except json.JSONDecodeError:
            return '{"error": "Malformed JSON arguments"}'

        if not isinstance(kwargs, dict):
            return json.dumps({"error": "Arguments must be a JSON object, e.g. {\"arg\": \"value\"}"})

        method = getattr(self.graph_context, name, None)
        if not method:
            return json.dumps({"error": f"Tool {name} not found"})

        try:
            result = method(**kwargs)
            return json.dumps(result)
        except Exception as e:
            logger.error("tool_execution_failed", tool=name, error=str(e))
            return json.dumps({"error": str(e)})

    def _positional_to_json(self, name: str, inner: str) -> str:
        """Maps positional args (e.g. get_path("RF1", "R37")) to a JSON object
        using the parameter order declared in TOOLS_SCHEMA."""
        params = _TOOL_PARAMS.get(name, [])
        raw = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
        return json.dumps(dict(zip(params, raw, strict=False)))

    def _parse_react_tool_call(self, text: str) -> tuple[str, str] | None:
        """Parses a text-form tool call across the formats local models emit.
        Returns (name, args_json). Gated on valid tool names to avoid matching
        prose. Handles three observed shapes:
          A) {"name": "tool", "arguments": {...}}
          B) tool_name({...}) / tool_name {...}   (object args, possibly prefixed)
          C) tool_name("a", "b")                  (positional args)
        """
        # Strategy A: explicit name/arguments JSON envelope.
        name_m = re.search(r'"name"\s*:\s*"([a-zA-Z0-9_]+)"', text)
        if name_m and name_m.group(1) in _VALID_TOOLS:
            name = name_m.group(1)
            args_m = re.search(r'"(?:arguments|parameters)"\s*:\s*(\{.*?\})', text, re.DOTALL)
            return name, args_m.group(1) if args_m else "{}"

        # Strategy B/C: locate the earliest valid tool name mentioned, then read
        # whichever comes first after it — an object {...} or a (positional) list.
        present = sorted((text.find(n), n) for n in _VALID_TOOLS if n in text)
        if not present:
            return None
        name = present[0][1]
        rest = text[text.find(name) + len(name):]

        obj_m = re.search(r"\{.*?\}", rest, re.DOTALL)
        pos_m = re.search(r"\(\s*([^){}]*?)\s*\)", rest, re.DOTALL)
        candidates: list[tuple[int, str, str]] = []
        if obj_m:
            candidates.append((obj_m.start(), "obj", obj_m.group(0)))
        if pos_m:
            candidates.append((pos_m.start(), "pos", pos_m.group(1)))
        if not candidates:
            return None
        candidates.sort()
        _, kind, payload = candidates[0]
        if kind == "obj":
            return name, payload
        return name, self._positional_to_json(name, payload)

    async def query(self, user_question: str) -> str:
        """Runs the agent loop to answer a user question."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_question}
        ]

        for iteration in range(self.max_iterations):
            logger.info("agent_iteration", iteration=iteration+1)
            response_msg = await self.llm_client.chat(messages, tools=TOOLS_SCHEMA)

            has_native_tools = bool(getattr(response_msg, "tool_calls", None))
            content = getattr(response_msg, "content", "") or ""

            # Record the assistant's message
            msg_dict: dict[str, Any] = {"role": "assistant"}
            if content:
                msg_dict["content"] = content
            if has_native_tools:
                # Need to convert objects back to dict for OpenAI API compatibility
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_msg.tool_calls
                ]
            messages.append(msg_dict)

            # Handle native tool calls
            if has_native_tools:
                for tool_call in response_msg.tool_calls:
                    name = tool_call.function.name
                    args_str = tool_call.function.arguments
                    logger.debug("executing_native_tool", name=name, args=args_str)

                    result_str = self._execute_tool(name, args_str)

                    if "Malformed JSON" in result_str:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": name,
                            "content": '{"error": "Previous tool_call had malformed JSON, please retry with valid JSON"}'
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": name,
                            "content": result_str
                        })
                continue  # Go to next iteration to let LLM process tool outputs

            # Handle ReAct-style text tool calls (Fallback)
            if content:
                react_call = self._parse_react_tool_call(content)
                if react_call:
                    name, args_str = react_call
                    logger.debug("executing_react_tool", name=name, args=args_str)

                    result_str = self._execute_tool(name, args_str)

                    # Instead of formal tool role (which requires tool_call_id),
                    # we just append a user message with the observation
                    if "Malformed JSON" in result_str:
                        messages.append({
                            "role": "user",
                            "content": 'Observation: {"error": "Previous tool_call had malformed JSON, please retry with valid JSON"}'
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"Observation: {result_str}"
                        })
                    continue

            # If no tools called (native or react), we are done
            self.last_messages = messages
            return content

        self.last_messages = messages
        return "Could not answer within max iterations."
