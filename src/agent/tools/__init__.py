"""Agent tools."""

from agent.tools.bookmarked_arxiv_paper import (
    get_bookmarked_arxiv_papers,
    search_bookmarked_arxiv_paper,
)
from agent.tools.arxiv_paper import recommend_todays_arxiv_papers

tools = [
    get_bookmarked_arxiv_papers,
    search_bookmarked_arxiv_paper,
    recommend_todays_arxiv_papers
]
tool_by_name = {tool.name: tool for tool in tools}
