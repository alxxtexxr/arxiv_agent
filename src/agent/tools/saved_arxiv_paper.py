"""arXiv paper tools and helpers."""

from pathlib import Path
from typing import Any

import arxiv
from langchain.tools import tool

from agent.utils import format_arxiv_paper

SAVED_ARXIV_LINKS_FILE = Path(__file__).parent.parent / "data" / "saved_arxiv_links.txt"

def _fetch_all() -> list[dict[str, Any]]:
    """Fetch all saved arXiv papers based on the links in the saved_arxiv_links.txt file."""
    
    with open(SAVED_ARXIV_LINKS_FILE) as f:
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
def get_saved_arxiv_papers() -> str:
    """Get saved arXiv papers."""
    
    papers = _fetch_all()
    if not papers:
        return "No saved arXiv papers found."

    return "\n\n".join(format_arxiv_paper(
        title=p["title"], 
        link=p["link"], 
        abstract=p["abstract"],
    ) for p in papers)

@tool
def search_saved_arxiv_paper(query: str) -> str:
    """Search saved arXiv papers based on a query."""
    
    papers = _fetch_all()
    if not papers:
        return "No saved arXiv papers found."
    
    query_lower = query.lower()
    matches = [
        p for p in papers
        if query_lower in p["title"].lower() or query_lower in p["abstract"].lower()
    ]
    if not matches:
        return f"No saved arXiv papers match '{query}'."

    return "\n\n".join(format_arxiv_paper(p) for p in matches)
