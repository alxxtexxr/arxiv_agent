"""Tool-calling node for the agent graph."""

from typing import Any, Dict

from langchain.messages import ToolMessage

from agent.state import State
from agent.tools import _fetch_saved_arxiv_papers, tool_by_name

def call_tool(state: State) -> Dict[str, Any]:
    """Call a tool and update state with fetched arXiv papers."""
    messages: list[ToolMessage] = []
    saved_arxiv_papers = state.get("saved_arxiv_papers", None)
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name in ("list_saved_arxiv_papers", "search_saved_arxiv_paper"):
            if saved_arxiv_papers is None:
                saved_arxiv_papers = _fetch_saved_arxiv_papers()
            tool_args["papers"] = saved_arxiv_papers

        tool = tool_by_name[tool_name]
        observation = tool.invoke(tool_args)
        messages.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {
        "messages": messages,
        "saved_arxiv_papers": saved_arxiv_papers,
    }
