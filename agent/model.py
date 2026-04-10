from typing import Any

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

from .config import Settings


@traceable(run_type="chain", name="Qwen Reasoning")
async def log_reasoning(reasoning: str) -> str:
    print(f"\n[Reasoning]\n{reasoning}\n")
    return reasoning


class LocalQwenModel:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = wrap_openai(
            AsyncOpenAI(
                base_url=settings.base_url,
                api_key=settings.api_key,
            )
        )

    async def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": self.settings.system_prompt},
                *messages,
            ],
            tools=tools,
        )

        message = response.choices[0].message
        reasoning = (message.model_extra or {}).get("reasoning_content", "")
        if reasoning:
            await log_reasoning(reasoning)

        return _assistant_message_to_dict(message)


def _assistant_message_to_dict(message: ChatCompletionMessage) -> dict[str, Any]:
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }
    tool_calls = _tool_calls_to_dict(message.tool_calls)
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    return assistant_message


def _tool_calls_to_dict(
    tool_calls: list[ChatCompletionMessageToolCall] | None,
) -> list[dict[str, Any]]:
    if not tool_calls:
        return []

    return [
        {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }
        for tool_call in tool_calls
    ]
