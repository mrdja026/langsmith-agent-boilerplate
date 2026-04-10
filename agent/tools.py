from typing import Any

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool


@tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "San Francisco": "Foggy, 62°F",
        "Tokyo": "Clear, 68°F",
    }
    return f"The weather in {city} is {weather_data.get(city, 'Unknown')}."


ALL_TOOLS = [get_weather]


def get_openai_tool_schemas() -> list[dict[str, Any]]:
    """Convert LangChain tools to OpenAI-format tool schemas."""
    return [convert_to_openai_tool(t) for t in ALL_TOOLS]
