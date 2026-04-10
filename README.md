# Local Qwen agent with LangGraph + LangSmith tracing

This project runs a **local Qwen agent** through a vLLM OpenAI-compatible endpoint and sends **traces only** to LangSmith.

# Bottstraped by Copilot Gpt-4 high reasoning for a plan with Opus 4.6 for followup fixes to apply Langsmith Docs for The Basic Agent Loop with one Tool

## 1. Install dependencies

```bash
uv sync
```

## 2. Configure environment

Create a `.env` file with your LangSmith settings:

```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=my-test-project
```

Optional local model settings:

```bash
LOCAL_QWEN_MODEL=Qwen/Qwen3.5-4B
LOCAL_OPENAI_BASE_URL=http://localhost:8000/v1
LOCAL_OPENAI_API_KEY=local-token
```

## 3. Start vLLM

```bash
uv run vllm serve Qwen/Qwen3.5-4B \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

## 4. Run the agent

### LangGraph dev / Studio-style workflow

```bash
uv run langgraph dev
```

This starts the LangGraph local development server using `langgraph.json`.

Use this mode when you want the LangGraph dev experience instead of the plain terminal loop. The repo exports the compiled graph from `agent/studio.py`, and environment variables are loaded from `.env`.

### Interactive multi-turn chat

```bash
uv run python main.py --thread-id demo-thread
```

### Single turn

```bash
uv run python main.py --thread-id demo-thread --message "What is the weather in San Francisco?"
```

## Project layout

- `main.py` - thin launcher for the local agent
- `agent/entry.py` - user-facing entrypoints
- `agent/graph.py` - LangGraph workflow and tool loop
- `agent/model.py` - local Qwen model call and reasoning capture
- `agent/tools.py` - tool registry and execution
- `agent/state.py` - graph state definition
- `agent/config.py` - runtime configuration
- `agent/studio.py` - exported graph for `langgraph dev`
- `langgraph.json` - LangGraph dev server configuration

## What shows up in LangSmith

When tracing is enabled, LangSmith should show:

- the graph run
- model-node execution
- tool-node execution
- individual tool spans
- the custom `Qwen Reasoning` span for `reasoning_content`

The model inference still runs locally on your machine; LangSmith only receives trace metadata and run structure.
