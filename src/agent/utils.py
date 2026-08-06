def format_arxiv_paper(title: str, link: str, abstract: str) -> str:
    """Format arXiv paper metadata into a readable string."""
    return (
        f"Title: {title}\n"
        f"Link: {link}\n"
        f"Abstract: {abstract}"
    )