import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

from agent.entry import chat_once, run_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Qwen agent against a vLLM OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--message",
        help="Run a single turn instead of starting the interactive multi-turn chat loop.",
    )
    parser.add_argument(
        "--thread-id",
        default="local-qwen-agent",
        help="Conversation thread identifier used by the LangGraph checkpointer.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.message:
        reply = await chat_once(args.message, thread_id=args.thread_id)
        print(f"Assistant: {reply}")
        return

    await run_cli(thread_id=args.thread_id)


if __name__ == "__main__":
    asyncio.run(main())
