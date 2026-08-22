"""Agent tools."""

from arxiv_agent.tools.bookmarked_arxiv_papers import (
    get_bookmarked_arxiv_papers,
    search_bookmarked_arxiv_papers,
)
from arxiv_agent.tools.bookmarked_arxiv_urls_from_github import extract_bookmarked_arxiv_urls_from_github
from arxiv_agent.tools.recommend_arxiv_papers import recommend_todays_arxiv_papers, recommend_arxiv_papers_by_date

tools = [
    get_bookmarked_arxiv_papers,
    search_bookmarked_arxiv_papers,
    recommend_todays_arxiv_papers,
    extract_bookmarked_arxiv_urls_from_github,
    recommend_arxiv_papers_by_date,
]
tool_by_name = {tool.name: tool for tool in tools}
