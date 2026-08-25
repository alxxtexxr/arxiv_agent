"""Bookmarked arXiv paper tools and helpers."""

from pathlib import Path
from typing import Any

import arxiv
from langchain.tools import tool

from arxiv_agent.utils import format_arxiv_paper

BOOKMARKED_ARXIV_DATA_DIR = Path(__file__).parent.parent / "data"


def _get_bookmarked_urls_files() -> list[Path]:
    """Return all bookmarked URL files, local file first, excluding examples."""
    files = [
        p for p in BOOKMARKED_ARXIV_DATA_DIR.glob("bookmarked_arxiv_urls*.txt") if "example" not in p.name
    ]
    return sorted(files, key=lambda p: (0 if p.name == "bookmarked_arxiv_urls.txt" else 1, p.name))


def _fetch_papers(with_abstract: bool=False) -> list[dict[str, Any]]:
    """Fetch bookmarked arXiv papers from all bookmark files."""
    files = _get_bookmarked_urls_files()
    if not files:
        return []

    urls: list[str] = []
    seen_ids: set[str] = set()
    for file in files:
        for line in file.read_text().splitlines():
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

# Test the function
if __name__ == "__main__":
    from pprint import pprint
    
    data = _fetch_papers(with_abstract=True)
    
    pprint(data)