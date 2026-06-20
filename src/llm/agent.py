import json
import re
from typing import Any

import openai
import structlog

from src.llm.tools import TOOLS_SCHEMA, GraphContext

logger = structlog.get_logger("agent")

class LLMClient:
    """Abstract interface for LLM clients."""
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Ollama client using OpenAI SDK compatibility layer."""
    def __init__(self, model: str = "llama3.1:8b-instruct-q4_K_M"):
        self.model = model
        self.client = openai.AsyncOpenAI(
            base_url="http://localhost:11434/v1",
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
        # Simple mock logic based on the last message
        last_msg = messages[-1]["content"] if messages[-1].get("content") else ""

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
        self.system_prompt = (
            "You are a schematic AI reasoner. Use the provided tools to answer questions "
            "about the schematic graph. Be concise and technical. Always cite component refs "
            "in your answers.\n"
            "Llama 3.1 function calling can be unreliable; dual-mode parsing handles both native "
            "tool_calls and ReAct-style text output. This makes the agent robust to model quirks "
            "without coupling to a specific backend. If you want to call a tool, you can either use "
            "the native tool call format, or output the text exactly like:\n"
            "TOOL_CALL: tool_name({\"arg1\": \"val1\"})\n"
        )

    def _execute_tool(self, name: str, arguments_str: str) -> str:
        """Executes a tool on GraphContext and returns the JSON result string."""
        try:
            kwargs = json.loads(arguments_str)
        except json.JSONDecodeError:
            return '{"error": "Malformed JSON arguments"}'

        method = getattr(self.graph_context, name, None)
        if not method:
            return json.dumps({"error": f"Tool {name} not found"})

        try:
            result = method(**kwargs)
            return json.dumps(result)
        except Exception as e:
            logger.error("tool_execution_failed", tool=name, error=str(e))
            return json.dumps({"error": str(e)})

    def _parse_react_tool_call(self, text: str) -> tuple[str, str] | None:
        """Parses a ReAct-style tool call from text. Returns (name, args_json)."""
        pattern = r"TOOL_CALL:\s*([a-zA-Z0-9_]+)\((.*?)\)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None

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
            return content

        return "Could not answer within max iterations."
