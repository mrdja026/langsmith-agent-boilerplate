import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    model: str
    base_url: str
    api_key: str
    system_prompt: str


@lru_cache
def get_settings() -> Settings:
    return Settings(
        model=os.getenv("LOCAL_QWEN_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen3.5-4B")),
        base_url=os.getenv(
            "LOCAL_OPENAI_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        ),
        api_key=os.getenv("LOCAL_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "local-token")),
        system_prompt=os.getenv(
            "AGENT_SYSTEM_PROMPT",
            "You are a helpful local assistant. Use tools when needed and answer clearly.",
        ),
    )
