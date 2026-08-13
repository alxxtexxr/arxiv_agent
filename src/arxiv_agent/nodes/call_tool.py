"""Tool-calling node for the agent graph."""

from typing import Any, Dict

from langgraph.graph import MessagesState
from langchain.messages import ToolMessage

from arxiv_agent.tools import tool_by_name

def call_tool(state: MessagesState) -> Dict[str, Any]:
    """Call the tool with the current state and return the new messages."""
    
    # Extract the tool calls from the last message in the state
    tool_calls = state["messages"][-1].tool_calls
    tool_call_names = [tool_call["name"] for tool_call in tool_calls]
    
    # Validate that only one tool call from the unique set is present
    unique_tool_names = set([
        "get_bookmarked_arxiv_papers", 
        "search_bookmarked_arxiv_paper", 
        "recommend_todays_arxiv_ai_papers",
    ])
    is_valid = sum(1 for item in tool_call_names if item in unique_tool_names) <= 1
    if not is_valid:
        return {"messages": [ToolMessage(
            content=f"Only one tool call from {unique_tool_names} is allowed per node. Found: {tool_call_names}", 
            tool_call_id=tool_calls[0]["id"],
        )]}
    
    # Call the tools and collect the messages
    messages: list[ToolMessage] = []
    for tool_call in tool_calls:
        tool = tool_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        messages.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        
    return {"messages": messages}
