"""Model-calling node for the agent graph."""

import os
from typing import Any, Dict

from langchain.messages import SystemMessage
from langgraph.runtime import Runtime

from agent.model import model_with_tools
from agent.state import Context, State

def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Call the model to decide whether to call a tool or respond."""
    return {
        "messages": [
            model_with_tools.invoke([
                SystemMessage(content=os.environ["SYSTEM_PROMPT"]),
            ] + state["messages"]),
        ],
        "num_model_calls": state.get("num_model_calls", 0) + 1,
    }
