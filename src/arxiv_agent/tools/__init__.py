"""Agent tools."""

# Lazy imports: heavy modules (recommend_arxiv_papers loads the embedding model)
# are imported on first access so that lightweight callers (e.g. the daily-job
# parent process) don't pull the model into memory.

def __getattr__(name):
    if name == "tools":
        return _get_tools()
    if name == "tool_by_name":
        return {t.name: t for t in _get_tools()}
    if name in ("recommend_todays_arxiv_papers", "recommend_arxiv_papers_by_date"):
        from arxiv_agent.tools import recommend_arxiv_papers
        return getattr(recommend_arxiv_papers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_tools():
    from arxiv_agent.tools.bookmarked_arxiv_papers import (
        bookmark_arxiv_papers,
        get_bookmarked_arxiv_papers,
        search_bookmarked_arxiv_papers,
        unbookmark_arxiv_papers,
    )
    from arxiv_agent.tools.bookmarked_arxiv_urls_from_github import (
        extract_bookmarked_arxiv_urls_from_github,
    )
    from arxiv_agent.tools.recommend_arxiv_papers import (
        recommend_arxiv_papers_by_date,
        recommend_todays_arxiv_papers,
    )

    return [
        get_bookmarked_arxiv_papers,
        search_bookmarked_arxiv_papers,
        bookmark_arxiv_papers,
        unbookmark_arxiv_papers,
        recommend_todays_arxiv_papers,
        extract_bookmarked_arxiv_urls_from_github,
        recommend_arxiv_papers_by_date,
    ]
