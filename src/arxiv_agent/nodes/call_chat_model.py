"""Chat model-calling node for the agent graph."""

from pathlib import Path
from time import time
from typing import Any, Dict

from langchain.messages import SystemMessage
from langgraph.graph import MessagesState

from arxiv_agent.models.chat_model import chat_model_with_tools

ACTIVITY_FILE = Path(__file__).parent.parent.parent / ".last_activity"

def _update_activity_timestamp() -> None:
    """Record the current time as the last activity."""
    ACTIVITY_FILE.write_text(str(time.time()))

SYSTEM_PROMPT = """You are a helpful assistant.

###### Guidelines
- If the user asks for paper recommendations vaguely, don't assume they want personalized recommendations based on their bookmarked papers.
  Instead, clarify whether they want:
  - Personalized recommendations based on their bookmarked papers, or
  - Query-based recommendations based on specific topics.
"""

def call_chat_model(state: MessagesState) -> Dict[str, Any]:
    """Call the chat model with the current state and return the new messages."""
    _update_activity_timestamp()
    return {
        "messages": [
            chat_model_with_tools.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
            ] + state["messages"]),
        ],
    }
