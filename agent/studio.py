from agent.config import get_settings
from agent.graph import build_agent_graph
from agent.model import LocalQwenModel

graph = build_agent_graph(LocalQwenModel(get_settings()), use_checkpointer=False)
