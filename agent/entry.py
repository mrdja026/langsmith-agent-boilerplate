from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .config import get_settings
from .graph import build_agent_graph
from .model import LocalQwenModel


class LocalQwenAgent:
    def __init__(self):
        settings = get_settings()
        self.graph = build_agent_graph(LocalQwenModel(settings))

    async def chat(self, user_message: str, thread_id: str) -> str:
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return _last_assistant_content(result["messages"])


async def chat_once(message: str, thread_id: str = "local-qwen-agent") -> str:
    agent = LocalQwenAgent()
    return await agent.chat(message, thread_id=thread_id)


async def run_cli(thread_id: str | None = None) -> None:
    active_thread_id = thread_id or str(uuid4())
    agent = LocalQwenAgent()

    print(f"Thread ID: {active_thread_id}")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        reply = await agent.chat(user_message, thread_id=active_thread_id)
        print(f"Assistant: {reply}")


def _last_assistant_content(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content or "")
    raise ValueError("Graph completed without an assistant message.")
