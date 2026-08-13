"""Agent tools."""

from arxiv_agent.tools.bookmarked_arxiv_paper import (
    get_bookmarked_arxiv_papers,
    search_bookmarked_arxiv_paper,
)
from arxiv_agent.tools.arxiv_paper import recommend_todays_arxiv_papers, recommend_arxiv_papers_by_date

tools = [
    get_bookmarked_arxiv_papers,
    search_bookmarked_arxiv_paper,
    recommend_todays_arxiv_papers,
    recommend_arxiv_papers_by_date,
]
tool_by_name = {tool.name: tool for tool in tools}
