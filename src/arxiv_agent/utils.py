def format_arxiv_paper(title: str, url: str, abstract: str) -> str:
    """Format arXiv paper metadata into a readable string."""
    return (
        f"Title: {title}\n"
        f"URL: {url}\n"
        f"Abstract: {abstract}"
    )