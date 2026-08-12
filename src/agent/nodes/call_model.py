"""Model-calling node for the agent graph."""

from typing import Any, Dict

from langgraph.graph import MessagesState
from langchain.messages import SystemMessage

from agent.model import model_with_tools

SYSTEM_PROMPT = """You are a helpful assistant.

###### Guidelines
- If the user asks for paper recommendations vaguely, don't assume they want personalized recommendations based on their bookmarked papers.
  Instead, clarify whether they want:
  - Personalized recommendations based on their bookmarked papers, or
  - Query-based recommendations based on specific topics.
"""

def call_model(state: MessagesState) -> Dict[str, Any]:
    """Call the model with the current state and return the new messages."""
    return {
        "messages": [
            model_with_tools.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
            ] + state["messages"]),
        ],
    }
