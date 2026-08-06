"""Routing logic for the agent graph."""

from typing import Literal

from agent.state import State

def call_tool_or_stop(state: State) -> Literal["call_tool", "__end__"]:
    """Determine whether to call a tool or stop."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    return "__end__"
