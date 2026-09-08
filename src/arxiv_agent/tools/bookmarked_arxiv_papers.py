"""Bookmarked arXiv paper tools and helpers."""

import os
import re
from pathlib import Path
from typing import Any

import arxiv
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv() # Load environment variables from .env file

BOOKMARKED_ARXIV_DATA_DIR = Path(__file__).parent.parent / "data"


def _get_bookmarked_files() -> list[Path]:
    """Return all bookmark files, main file first, excluding examples."""
    files = [
        p for p in BOOKMARKED_ARXIV_DATA_DIR.glob("bookmarked_arxiv_urls*.txt") if "example" not in p.name
    ]
    return sorted(files, key=lambda p: (0 if p.name == "bookmarked_arxiv_urls.txt" else 1, p.name))


def _fetch_papers(with_abstract: bool=False) -> list[dict[str, Any]]:
    """Fetch bookmarked arXiv papers from all bookmark files."""
    bookmark_files = _get_bookmarked_files()
    if not bookmark_files:
        return []

    urls: list[str] = []
    seen_ids: set[str] = set()
    for bookmark_file in bookmark_files:
        for line in bookmark_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            paper_id = line.rstrip("/").split("/")[-1].replace(".pdf", "")
            if paper_id not in seen_ids:
                seen_ids.add(paper_id)
                urls.append(line)

    paper_ids = [url.rstrip("/").split("/")[-1].replace(".pdf", "") for url in urls]

    client = arxiv.Client()
    search = arxiv.Search(id_list=paper_ids)

    papers = []
    for r in client.results(search):
        paper = {
            "title": r.title,
            "url": r.entry_id,
        }
        if with_abstract:
            paper["abstract"] = r.summary
        papers.append(paper)
    return papers


def format_arxiv_paper(title: str, url: str, abstract: str | None = None) -> str:
    """Format arXiv paper metadata into a readable string."""
    if abstract is not None:
        return (
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Abstract: {abstract}"
        )
    return f"Title: {title}\nURL: {url}"


@tool
def get_bookmarked_arxiv_papers(with_abstract: bool=False) -> str:
    """Get bookmarked arXiv papers.

    Args:
        with_abstract: Whether to include abstracts in the output. Defaults to False.
    
    Returns:
        A string containing the bookmarked arXiv papers.
    """
    papers = _fetch_papers(with_abstract)
    if not papers:
        return "No bookmarked arXiv papers found."

    return "\n\n".join(format_arxiv_paper(
        title=p["title"], 
        url=p["url"], 
        abstract=p["abstract"] if with_abstract else None,
    ) for p in papers)

@tool
def search_bookmarked_arxiv_papers(query: str) -> str:
    """Search for bookmarked arXiv papers by title or abstract.

    Args:
        query: The query to search for.
    
    Returns:
        A string containing the search results.
    """
    papers = _fetch_papers(with_abstract=True)
    if not papers:
        return "No bookmarked arXiv papers found."
    
    query_lower = query.lower()
    matches = [
        p for p in papers
        if query_lower in p["title"].lower() or query_lower in p["abstract"].lower()
    ]
    if not matches:
        return f"No bookmarked arXiv papers match '{query}'."

    return "\n\n".join(format_arxiv_paper(title=p["title"], url=p["url"], abstract=p["abstract"]) for p in matches)

def is_valid_arxiv_abs_url(url: str) -> bool:
    """Checks if a URL matches the format of an arXiv abstract URL: https://arxiv.org/abs/{4 digits}.{5 digits}"""
    pattern = r"^https://arxiv\.org/abs/\d{4}\.\d{5}$"
    return bool(re.match(pattern, url))

@tool
def bookmark_arxiv_papers(urls: list[str], secret_password: str) -> str:
    """Bookmark one or more arXiv papers by URL.

    Args:
        urls: A list of URLs of the arXiv papers to bookmark.
        secret_password: A secret password for authentication.

    Returns:
        A string indicating the result of the bookmarking operation.
    """
    # Check the secret password
    if secret_password != os.environ["BOOKMARK_SECRET_PASSWORD"]:
        return "Invalid secret password."
    
    # Get the list of bookmark files and the main bookmark file
    bookmark_files = _get_bookmarked_files()
    main_bookmark_file = bookmark_files[0]

    # Read existing bookmarked URLs from all bookmark files
    existing_urls = []
    for bookmark_file in bookmark_files:
        with open(bookmark_file, "r") as f:
            existing_urls.extend(f.read().splitlines())

    new_urls, responses = [], []
    for url in urls:
        # Sanitize the URL
        url = url.replace(".pdf", "")   # Remove .pdf extension if present
        url = url if url.count("v") == 1 else url.rsplit("v", 1)[0] # Remove version number if present
        url = url.replace("pdf", "abs")
        url = url.replace("html", "abs")
        url = url.replace("src", "abs")

        # Skip if the URL is not a valid arXiv URL
        if not is_valid_arxiv_abs_url(url):
            responses.append(
                f"The URL '{url}' is not a valid arXiv abstract URL. "
                f"Please provide a valid URL in the format 'https://arxiv.org/abs/{{4 digits}}.{{5 digits}}'."
            )
            continue

        # Skip if the URL is already bookmarked
        if url in existing_urls:
            responses.append(f"The paper at {url} is already bookmarked.")
            continue

        new_urls.append(url)
        responses.append(f"Successfully bookmarked the paper at {url}.")
    
    assert len(new_urls) == len(responses) - (len(urls) - len(new_urls)), "Mismatch in new URLs and responses count."
        
    # Append new URLs to the main bookmark file
    with open(main_bookmark_file, "a") as f:
        for url in new_urls:
            f.write(url + "\n")

    return "\n".join(responses)
        
@tool
def unbookmark_arxiv_papers(urls: list[str], secret_password: str) -> str:
    """Unbookmark one or more arXiv papers by URL.

    Args:
        urls: A list of URLs of the arXiv papers to unbookmark.
        secret_password: A secret password for authentication.

    Returns:
        A string indicating the result of the unbookmarking operation.
    """
    # Check the secret password
    if secret_password != os.environ["BOOKMARK_SECRET_PASSWORD"]:
        return "Invalid secret password."

    # Get the list of bookmark files and the main bookmark file
    bookmark_files = _get_bookmarked_files()
    main_bookmark_file = bookmark_files[0]
    
    # Read existing bookmarked URLs from all bookmark files
    existing_urls, source_files = [], []
    for bookmark_file in bookmark_files:
        with open(bookmark_file, "r") as f:
            file_urls = f.read().splitlines()
            existing_urls.extend(file_urls)
            source_files.extend([bookmark_file] * len(file_urls))

    assert len(existing_urls) == len(source_files), "Mismatch in existing URLs and source files count."

    kept_urls, responses = [], []
    for i, existing_url in enumerate(existing_urls):
        if existing_url in urls:
            # Don't bookmark the URL if it's not in the main bookmark file
            if source_files[i] == main_bookmark_file:
                responses.append(f"Successfully unbookmarked the paper at {existing_url}.")
            else:
                responses.append(f"The paper at {existing_url} is not in the main bookmark file and cannot be unbookmarked.")
        else:
            # Keep the URL
            kept_urls.append(existing_url)

    # Write the kept URLs back to the main bookmark file
    with open(main_bookmark_file, "w") as f:
        for url in kept_urls:
            f.write(url + "\n")

    return "\n".join(responses)

# Test the function
if __name__ == "__main__":
    from pprint import pprint
    
    # data = _fetch_papers(with_abstract=True)
    # pprint(data)
    
    data = bookmark_arxiv_papers.invoke(input={
        "urls": [
            "https://arxiv.org/abs/2306.00001",
            "https://arxiv.org/abs/2306.00002v2",
        ],
        "secret_password": os.environ["BOOKMARK_SECRET_PASSWORD"],
    })
    pprint(data)
    
    data = unbookmark_arxiv_papers.invoke(input={
        "urls": [
            "https://arxiv.org/abs/2306.00001",
            "https://arxiv.org/abs/2306.00002",
        ],
        "secret_password": os.environ["BOOKMARK_SECRET_PASSWORD"],
    })
    pprint(data)