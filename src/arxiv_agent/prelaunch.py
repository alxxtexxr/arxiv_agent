"""Pre-launch steps for the arXiv agent.

Runs daily on EC2 boot: extracts bookmarks, syncs today's papers,
then stops the instance when done.
"""

import sys
import subprocess
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


def _is_today_synced() -> bool:
    """Check if today's papers are already embedded."""
    from arxiv_agent.tools.recommend_arxiv_papers import _has_current_chunks
    from datetime import date as date_cls

    return _has_current_chunks(date_cls.today().isoformat())


def _is_today_bookmarked() -> bool:
    """Check if today's bookmarks file already exists."""
    from pathlib import Path
    from arxiv_agent.tools.bookmarked_arxiv_urls_from_github import (
        _derive_github_bookmarks_name,
    )
    import os

    url = os.environ.get("GITHUB_BOOKMARKS_URL", "")
    if not url:
        return True  # No bookmarks configured, skip
    name = _derive_github_bookmarks_name(url)
    path = (
        Path(__file__).parent
        / "data"
        / f"bookmarked_arxiv_urls_from_github_{name}.txt"
    )
    return path.exists()


def run_daily_job() -> str:
    """Run the daily pre-launch job: bookmarks + embedding.

    Skips each step if already done today.
    Returns a summary of what was executed.
    """
    results = []

    if not _is_today_bookmarked():
        results.append(extract_bookmarks())
    else:
        results.append("Bookmarks already synced for today.")

    if not _is_today_synced():
        results.append(sync_today())
    else:
        results.append("Papers already embedded for today.")

    return " | ".join(results)


def stop_instance() -> None:
    """Stop the EC2 instance via the instance-control-api."""
    import os
    import requests

    api_url = (
        "https://instance-control-api.alimtegar404.workers.dev"
        "/v1/instances/arxiv-agent/stop"
    )
    api_key = os.environ.get("INSTANCE_CONTROL_API_KEY", "")
    headers = {"X-Api-Key": api_key} if api_key else {}
    try:
        requests.post(api_url, headers=headers, timeout=30)
        print("Instance stop requested.")
    except Exception as e:
        print(f"Failed to stop instance: {e}")


STEPS = {
    "extract_bookmarks": extract_bookmarks,
    "sync_today": sync_today,
    "daily_job": run_daily_job,
}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in STEPS:
        sys.exit(f"unknown step '{step}'; expected one of: {', '.join(STEPS)}")

    result = STEPS[step]()
    print(result)

    # After running the daily job, stop the instance
    if step == "daily_job":
        stop_instance()
