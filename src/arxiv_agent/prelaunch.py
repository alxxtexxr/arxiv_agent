"""Pre-launch steps for the arXiv agent.

Runs daily on EC2 boot: extracts bookmarks, syncs today's papers,
then stops the instance when done.

A single daily-job-done flag file (data/daily_job_done) records the
last completed date, preventing auto-stop on same-day re-boots.
"""

import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

from arxiv_agent.tools.bookmarked_arxiv_urls_from_github import (
    extract_bookmarked_arxiv_urls_from_github,
)

DATA_DIR = Path(__file__).parent / "data"
_DAILY_JOB_DONE_FLAG = DATA_DIR / "daily_job_done"


def _is_daily_job_done() -> bool:
    """Return True if the daily job has already completed today."""
    if not _DAILY_JOB_DONE_FLAG.exists():
        return False
    saved = _DAILY_JOB_DONE_FLAG.read_text().strip()
    return saved == date.today().isoformat()


def _mark_daily_job_done() -> None:
    """Write today's date to the daily-job-done flag file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _DAILY_JOB_DONE_FLAG.write_text(date.today().isoformat())


def _touch_activity() -> None:
    """Touch the activity file so the idle-stop cron script knows we're busy."""
    import time

    activity_file = Path(__file__).parent / ".last_activity"
    activity_file.write_text(str(time.time()))


def extract_bookmarks() -> str:
    """Refresh bookmarked arXiv URLs from the configured source."""
    return extract_bookmarked_arxiv_urls_from_github.invoke({})  # type: ignore[reportFunctionMemberAccess]


def sync_today() -> str:
    """Fetch, chunk, and embed today's arXiv papers.

    Runs in a subprocess so that all memory (embedding model, intermediate
    tensors) is released when the subprocess exits — keeping peak RSS lower.
    """
    today = date.today().isoformat()
    cmd = [
        sys.executable,
        "-m",
        "arxiv_agent.tools.recommend_arxiv_papers",
        today,
    ]
    logging.info("Spawning embedding subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return f"Synced papers for {today}."


def run_daily_job() -> tuple[str, bool]:
    """Run the daily pre-launch job: bookmarks + embedding.

    If the daily job has already completed today (flag file exists),
    skips everything and returns (summary, False) so the caller
    knows NOT to stop the instance.

    On first run of the day, executes each step, writes the
    done-flag, and returns (summary, True) so the caller can
    stop the instance.
    """
    if _is_daily_job_done():
        logging.info("Skipping — daily job already completed today.")
        return "Daily job already completed today. Instance will keep running.", False

    # Touch the activity file so the idle-stop cron script knows we're busy.
    _touch_activity()

    logging.info("Starting daily job...")
    results = []

    results.append(extract_bookmarks())
    results.append(sync_today())

    _mark_daily_job_done()
    logging.info("Daily job completed.")

    return " | ".join(results), True


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
        # pi-lens-ignore: python-ssrf
        requests.post(api_url, headers=headers, timeout=30)  # noqa: S501 — intentional allowlist
        logging.info("Instance stop requested.")
    except Exception as e:
        logging.error("Failed to stop instance: %s", e)


STEPS = {
    "extract_bookmarks": extract_bookmarks,
    "sync_today": sync_today,
    "daily_job": run_daily_job,
}

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in STEPS:
        sys.exit(f"unknown step '{step}'; expected one of: {', '.join(STEPS)}")

    result = STEPS[step]()
    if isinstance(result, tuple):
        message, did_work = result
    else:
        message, did_work = result, False
    logging.info("%s", message)

    # Only stop the instance on the first daily-job run of the day.
    # Subsequent boots same day find the flag file and skip the stop.
    if step == "daily_job" and did_work:
        stop_instance()
