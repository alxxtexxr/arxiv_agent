"""Bookmarked arXiv paper tools and helpers."""

from pathlib import Path
from typing import Any

import arxiv
from langchain.tools import tool

from arxiv_agent.utils import format_arxiv_paper

BOOKMARKED_ARXIV_URLS_FILE = Path(__file__).parent.parent / "data" / "bookmarked_arxiv_urls.txt"

def _fetch_papers() -> list[dict[str, Any]]:
    """Fetch bookmarked arXiv papers from the links file."""
    with open(BOOKMARKED_ARXIV_URLS_FILE) as f:
        links = [line.strip() for line in f if line.strip()]

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
