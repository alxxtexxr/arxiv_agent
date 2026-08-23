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


def _fetch_papers() -> list[dict[str, Any]]:
    """Fetch bookmarked arXiv papers from all bookmark files."""
    files = _get_bookmarked_urls_files()
    if not files:
        return []

    links: list[str] = []
    seen_ids: set[str] = set()
    for file in files:
        for line in file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            paper_id = line.rstrip("/").split("/")[-1].replace(".pdf", "")
            if paper_id not in seen_ids:
                seen_ids.add(paper_id)
                links.append(line)

    paper_ids = [link.rstrip("/").split("/")[-1].replace(".pdf", "") for link in links]

    client = arxiv.Client()
    search = arxiv.Search(id_list=paper_ids)

    papers = [
        {
            "title": r.title,
            "link": f"https://arxiv.org/abs/{r.entry_id}",
            "abstract": r.summary,
        }
        for r in client.results(search)
    ]

    return papers

@tool
def get_bookmarked_arxiv_papers() -> str:
    """Get all bookmarked arXiv papers."""
    papers = _fetch_papers()
    if not papers:
        return "No bookmarked arXiv papers found."

    return "\n\n".join(format_arxiv_paper(
        title=p["title"], 
        link=p["link"], 
        abstract=p["abstract"],
    ) for p in papers)

@tool
def search_bookmarked_arxiv_papers(query: str) -> str:
    """Search bookmarked arXiv papers based on a query."""
    papers = _fetch_papers()
    if not papers:
        return "No bookmarked arXiv papers found."
    
    query_lower = query.lower()
    matches = [
        p for p in papers
        if query_lower in p["title"].lower() or query_lower in p["abstract"].lower()
    ]
    if not matches:
        return f"No bookmarked arXiv papers match '{query}'."

    return "\n\n".join(format_arxiv_paper(p) for p in matches)

# Test the function
if __name__ == "__main__":
    from pprint import pprint
    
    data = _fetch_papers()
    
    pprint(data)