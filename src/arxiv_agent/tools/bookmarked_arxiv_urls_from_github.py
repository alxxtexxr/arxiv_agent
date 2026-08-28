"""Bookmarked arXiv URLs fetched from a GitHub file."""

import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()  # Load environment variables from .env file

ARXIV_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/([\d.]+)(?:v\d+)?(?:\.pdf)?"
)


def _derive_github_bookmarks_name(url: str) -> str:
    """Derive a normalized name for the bookmarks source from its URL.

    URL-decodes the last path segment, strips the file extension, lowercases
    it, and sanitizes it to filesystem-safe characters.
    """
    filename = unquote(urlparse(url).path.rstrip("/").split("/")[-1])
    stem = filename.rsplit(".", 1)[0]
    name = re.sub(r"[^a-z0-9_-]+", "_", stem.lower()).strip("_")
    return name or "bookmarks"


@tool
def extract_bookmarked_arxiv_urls_from_github() -> str:
    """Extract arXiv URLs from the configured bookmarks source and store them locally.

    Fetches the bookmarks source, extracts every arXiv URL from its content
    (normalized to ``https://arxiv.org/abs/{id}`` form and deduplicated), and
    persists them for later use. Returns a confirmation message.
    """
    url = os.environ["GITHUB_BOOKMARKS_URL"]
    token = os.environ["GITHUB_TOKEN"]
    
    if not url or not token:
        return "Missing GITHUB_BOOKMARKS_URL or GITHUB_TOKEN environment variables."

    response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()

    urls = []
    for match in ARXIV_URL_PATTERN.finditer(response.text):
        arxiv_id = match.group(1)
        canonical = f"https://arxiv.org/abs/{arxiv_id}"
        if canonical not in urls:
            urls.append(canonical)

    if urls:
        name = _derive_github_bookmarks_name(url)
        output_path = (
            Path(__file__).parent.parent / "data" / f"bookmarked_arxiv_urls_from_github_{name}.txt"
        )
        output_path.write_text("\n".join(urls) + "\n")

        return f"Stored {len(urls)} arXiv URLs."
    else:
        return "No arXiv URLs found."


# Test the function
if __name__ == "__main__":
    from pprint import pprint

    pprint(extract_bookmarked_arxiv_urls_from_github.invoke({}))
