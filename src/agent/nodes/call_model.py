"""Model-calling node for the agent graph."""

from typing import Any, Dict

from langgraph.graph import MessagesState
from langchain.messages import SystemMessage

from agent.model import model_with_tools

SYSTEM_PROMPT = "You are a helpful assistant."

def call_model(state: MessagesState) -> Dict[str, Any]:
    """Call the model with the current state and return the new messages."""
    return {
        "messages": [
            model_with_tools.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
            ] + state["messages"]),
        ],
    }
