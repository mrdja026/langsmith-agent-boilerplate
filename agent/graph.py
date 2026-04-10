import json
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .model import LocalQwenModel
from .state import AgentState
from .tools import ALL_TOOLS, get_openai_tool_schemas


def _to_openai_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages to OpenAI-format dicts for the API call."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.type == "human":
            result.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            d: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["args"])
                                if isinstance(tc["args"], dict)
                                else tc["args"]
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(d)
        elif msg.type == "tool":
            assert isinstance(msg, ToolMessage)
            result.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name or "",
                }
            )
        elif msg.type == "system":
            result.append({"role": "system", "content": msg.content})
    return result


def build_agent_graph(model: LocalQwenModel, *, use_checkpointer: bool = True):
    openai_schemas = get_openai_tool_schemas()

    async def model_node(state: AgentState) -> dict[str, list[dict[str, object]]]:
        openai_messages = _to_openai_messages(state["messages"])
        assistant_message = await model.call(
            messages=openai_messages,
            tools=openai_schemas,
        )
        return {"messages": [assistant_message]}

    builder = StateGraph(AgentState)
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    if use_checkpointer:
        return builder.compile(checkpointer=InMemorySaver())
    return builder.compile()
