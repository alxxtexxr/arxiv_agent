def format_arxiv_paper(title: str, url: str, abstract: str | None = None) -> str:
    """Format arXiv paper metadata into a readable string."""
    if abstract is not None:
        return (
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Abstract: {abstract}"
        )
    return f"Title: {title}\nURL: {url}"