"""Pre-launch steps for the arXiv agent's local launcher."""

import sys
from datetime import date

from arxiv_agent.tools.bookmarked_arxiv_urls_from_github import (
    extract_bookmarked_arxiv_urls_from_github,
)
from arxiv_agent.tools.recommend_arxiv_papers import _ensure_synced


def extract_bookmarks() -> str:
    """Refresh bookmarked arXiv URLs from the configured source."""
    return extract_bookmarked_arxiv_urls_from_github.invoke({})


def sync_today() -> str:
    """Fetch, chunk, and embed today's arXiv papers."""
    today = date.today().isoformat()
    _ensure_synced(today)
    return f"Synced papers for {today}."


STEPS = {"extract_bookmarks": extract_bookmarks, "sync_today": sync_today}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in STEPS:
        sys.exit(f"unknown step '{step}'; expected one of: {', '.join(STEPS)}")
    sys.stdout.write(f"{STEPS[step]()}\n")
