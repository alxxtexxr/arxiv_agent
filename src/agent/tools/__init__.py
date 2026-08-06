"""Agent tools."""

from agent.tools.saved_arxiv_paper import (
    get_saved_arxiv_papers,
    search_saved_arxiv_paper,
)
from agent.tools.ai_arxiv_paper import recommend_todays_arxiv_ai_papers

tools = [
    get_saved_arxiv_papers,
    search_saved_arxiv_paper,
    recommend_todays_arxiv_ai_papers
]
tool_by_name = {tool.name: tool for tool in tools}
