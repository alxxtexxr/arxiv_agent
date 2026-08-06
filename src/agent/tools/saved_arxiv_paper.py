"""arXiv paper tools and helpers."""

import os
from typing import Any

import arxiv
from langchain.tools import tool

def _fetch_saved_arxiv_papers() -> list[dict[str, Any]]:
    """Fetch arXiv paper metadata for all saved links."""
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "saved_arxiv_links.txt")
    with open(filepath) as f:
        links = [line.strip() for line in f if line.strip()]

    paper_ids = [link.rstrip("/").split("/")[-1].replace(".pdf", "") for link in links]

    client = arxiv.Client()
    search = arxiv.Search(id_list=paper_ids)

    papers = [
        {
            "entry_id": r.entry_id,
            "updated": r.updated.isoformat(),
            "published": r.published.isoformat(),
            "title": r.title,
            "authors": [author.name for author in r.authors],
            "summary": r.summary,
            "comment": r.comment,
            "journal_ref": r.journal_ref,
            "doi": r.doi,
            "primary_category": r.primary_category,
            "categories": r.categories,
            "pdf_url": r.pdf_url,
        }
        for r in client.results(search)
    ]

    return papers

def _format_arxiv_paper(paper: dict[str, Any]) -> str:
    """Format a single arXiv paper result into a readable string."""
    return (
        f"Title: {paper['title']}\n"
        f"Link: https://arxiv.org/abs/{paper['entry_id']}\n"
        f"Abstract: {paper['summary']}"
    )

@tool
def get_saved_arxiv_papers(papers: list[dict[str, Any]]) -> str:
    """Get a formatted list of saved arXiv papers.

    Args:
        papers: A list of saved arXiv paper dictionaries, populated automatically.

    Returns:
        A formatted list of saved arXiv papers, or a message if none are found."""
    if not papers:
        return "No saved arXiv papers found."

    return "\n\n".join(_format_arxiv_paper(p) for p in papers)

@tool
def search_saved_arxiv_paper(papers: list[dict[str, Any]], query: str) -> str:
    """Search for saved arXiv papers based on a query in titles and abstracts.

    Args:
        papers: A list of saved arXiv paper dictionaries, populated automatically.
        query: The query to search for in titles and abstracts.

    Returns:
        A formatted list of matching papers, or a message if none match.
    """
    query_lower = query.lower()
    matches = [
        p for p in papers
        if query_lower in p["title"].lower() or query_lower in p["summary"].lower()
    ]

    if not matches:
        return f"No saved arXiv papers match '{query}'."

    return "\n\n".join(_format_arxiv_paper(p) for p in matches)
