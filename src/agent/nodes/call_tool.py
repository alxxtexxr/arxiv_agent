"""Tool-calling node for the agent graph."""

from typing import Any, Dict

from langgraph.graph import MessagesState
from langchain.messages import ToolMessage

from agent.tools import tool_by_name

def call_tool(state: MessagesState) -> Dict[str, Any]:
    """Call the tool with the current state and return the new messages."""
    messages: list[ToolMessage] = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tool_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        messages.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": messages}
