"""Agent tools."""

from agent.tools.saved_arxiv_paper import (
    _fetch_saved_arxiv_papers,
    get_saved_arxiv_papers,
    search_saved_arxiv_paper,
)
from agent.tools.blog import search_blog_posts
from agent.tools.ai_arxiv_paper import recommend_todays_arxiv_ai_papers

tools = [
    get_saved_arxiv_papers,
    search_saved_arxiv_paper,
    search_blog_posts,
    recommend_todays_arxiv_ai_papers
]
tool_by_name = {tool.name: tool for tool in tools}
