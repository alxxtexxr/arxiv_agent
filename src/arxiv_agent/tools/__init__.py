"""Agent tools.

Heavy modules (recommend_arxiv_papers loads the embedding model) are
imported lazily so that lightweight callers (e.g. the daily-job parent
process) don't pull the model into memory at import time.
"""

from arxiv_agent.tools.bookmarked_arxiv_papers import (
    bookmark_arxiv_papers,
    get_bookmarked_arxiv_papers,
    search_bookmarked_arxiv_papers,
    unbookmark_arxiv_papers,
)
from arxiv_agent.tools.bookmarked_arxiv_urls_from_github import (
    extract_bookmarked_arxiv_urls_from_github,
)

# Eager list — only lightweight tools.  Heavy recommendation tools are
# appended lazily on first access of `tools` or `tool_by_name`.
_LIGHTWEIGHT_TOOLS = [
    get_bookmarked_arxiv_papers,
    search_bookmarked_arxiv_papers,
    bookmark_arxiv_papers,
    unbookmark_arxiv_papers,
    extract_bookmarked_arxiv_urls_from_github,
]


def _load_heavy_tools():
    from arxiv_agent.tools.recommend_arxiv_papers import (
        recommend_arxiv_papers_by_date,
        recommend_todays_arxiv_papers,
    )
    return recommend_todays_arxiv_papers, recommend_arxiv_papers_by_date


def _all_tools():
    t1, t2 = _load_heavy_tools()
    return [
        *_LIGHTWEIGHT_TOOLS,
        t1,
        t2,
    ]


# Module-level __getattr__ so `from arxiv_agent.tools import tools` and
# `from arxiv_agent.tools import tool_by_name` keep working.
def __getattr__(name):
    if name == "tools":
        return _all_tools()
    if name == "tool_by_name":
        return {t.name: t for t in _all_tools()}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
