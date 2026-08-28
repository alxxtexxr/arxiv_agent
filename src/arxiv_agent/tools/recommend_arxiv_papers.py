"""arXiv paper recommendation tool.

Fetches arXiv papers for a given date (today's RSS feed, or any past date via
the arXiv API), stores the chunks and their bge-m3 embeddings in PostgreSQL
(pgvector), and recommends papers via retrieval plus cross-encoder reranking.
"""

import hashlib
import os
import re
from builtins import sorted
from datetime import date as date_cls
from datetime import datetime
from typing import Any, Literal

import arxiv as arxiv_api
import feedparser
from anyio.functools import lru_cache
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from arxiv_agent import db
from arxiv_agent.utils import format_arxiv_paper

load_dotenv()

CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = os.environ["EMBEDDING_MODEL"]
RERANKER_MODEL_NAME = os.environ["RERANKER_MODEL"]
MAX_RECOMMENDATIONS = 5
MAX_BOOKMARKS_PROCESSED = 10

# Stored chunks are keyed to the configuration that produced them; changing
# any of the covered knobs invalidates stored embeddings, and the next access
# of a date re-syncs it with the new configuration.
EMBEDDING_VERSION = hashlib.sha1(
    f"{CHUNK_SIZE}:{CHUNK_OVERLAP}:{EMBEDDING_MODEL_NAME}".encode()
).hexdigest()[:12]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

_db_ready = False
_synced_dates: set[str] = set()


def _ensure_db_ready() -> None:
    """Provision the schema once, on first use."""
    global _db_ready
    if not _db_ready:
        db.init_schema()
        _db_ready = True


@lru_cache(maxsize=1)
def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the shared embedding model instance."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_reranker_model() -> HuggingFaceCrossEncoder:
    """Return the shared reranker model instance."""
    return HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)


def _normalize_arxiv_id(raw_id: str) -> str:
    """Normalize a paper identifier by stripping any version suffix."""
    return re.sub(r"v\d+$", "", raw_id.rstrip("/").split("/")[-1].replace(".pdf", ""))


def _fetch_feed_entries() -> list[dict[str, str]]:
    """Fetch today's papers from the arXiv RSS feed."""
    feed = feedparser.parse(f"https://rss.arxiv.org/rss/{os.environ['ARXIV_RECOMMENDATION_CATEGORY']}")
    return [
        {
            "arxiv_id": _normalize_arxiv_id(entry.link),
            "title": entry.title,
            "url": entry.link,
            "abstract": entry.summary.split("Abstract:", 1)[-1].strip(),
        }
        for entry in feed.entries
    ]


def _fetch_api_entries_for_date(target_date: str) -> list[dict[str, str]]:
    """Fetch papers for a specific date via the arXiv API.

    Uses ``submittedDate`` (original submission date), the only reliable
    date field on the legacy export API: ``announced_date_first`` caps at a
    fixed partial set (~36 entries regardless of date) and ``published``
    reflects original submission, not announcement. The queried semantic is
    therefore "papers originally submitted on date X".

    The legacy API paginates inconsistently across calls, so entries are
    merged across repeated attempts (keyed by arxiv id) until the result set
    stabilizes.
    """
    compact_date = target_date.replace("-", "")
    query = (
        f"cat:{os.environ['ARXIV_RECOMMENDATION_CATEGORY']} AND "
        f"submittedDate:[{compact_date}0000 TO {compact_date}2359]"
    )
    search = arxiv_api.Search(
        query=query,
        max_results=2000,
        sort_by=arxiv_api.SortCriterion.SubmittedDate,
        sort_order=arxiv_api.SortOrder.Ascending,
    )

    merged: dict[str, dict[str, str]] = {}
    previous_count = -1
    for _ in range(3):
        for result in arxiv_api.Client().results(search):
            merged[_normalize_arxiv_id(result.entry_id)] = {
                "arxiv_id": _normalize_arxiv_id(result.entry_id),
                "title": result.title,
                "url": f"https://arxiv.org/abs/{result.entry_id}",
                "abstract": result.summary,
            }
        if len(merged) == previous_count:
            break
        previous_count = len(merged)
    return list(merged.values())


def _sync_date(target_date: str) -> int:
    """Fetch, chunk, embed, and upsert a date's papers. Returns chunk count."""
    if target_date == date_cls.today().isoformat():
        entries = _fetch_feed_entries()
    else:
        entries = _fetch_api_entries_for_date(target_date)
    if not entries:
        return 0

    paper_docs = [
        format_arxiv_paper(title=e["title"], url=e["url"], abstract=e["abstract"])
        for e in entries
    ]
    chunks_per_paper = [text_splitter.split_text(doc) for doc in paper_docs]
    flat_chunks = [chunk for chunks in chunks_per_paper for chunk in chunks]
    flat_embeddings = _get_embedding_model().embed_documents(flat_chunks)

    rows = []
    offset = 0
    for paper_index, entry in enumerate(entries):
        for chunk_idx, chunk_text in enumerate(chunks_per_paper[paper_index]):
            rows.append({
                "arxiv_id": entry["arxiv_id"],
                "chunk_idx": chunk_idx,
                "title": entry["title"],
                "url": entry["url"],
                "abstract": entry["abstract"],
                "content": chunk_text,
                "embedding": flat_embeddings[offset],
            })
            offset += 1

    db.sync_date(target_date, rows, EMBEDDING_VERSION)
    return len(rows)


def _ensure_synced(target_date: str) -> None:
    """Ensure a date's papers are stored with the current embedding version.

    A date is re-synced when it was never synced, when the stored embedding
    version differs from the current one (chunking or embedding model
    changed), or after a transient fetch failure that left no papers behind.
    """
    _ensure_db_ready()
    if target_date in _synced_dates:
        return
    if db.get_sync_version(target_date) != EMBEDDING_VERSION:
        _sync_date(target_date)
    if db.date_has_chunks(target_date):
        _synced_dates.add(target_date)


class _ArxivDateRetriever(BaseRetriever):
    """Retrieves chunks of papers published on a specific date."""

    date: str
    k: int = 10

    def _get_relevant_documents(self, query: str) -> list[Document]:
        query_vector = _get_embedding_model().embed_query(query)
        rows = db.search_by_date(self.date, query_vector, self.k)
        return [
            Document(
                page_content=row["content"],
                metadata={
                    "arxiv_id": row["arxiv_id"],
                    "chunk_idx": row["chunk_idx"],
                    "title": row["title"],
                    "url": row["url"],
                    "abstract": row["abstract"],
                },
            )
            for row in rows
        ]


def _get_compression_retriever(target_date: str, top_n: int = 5) -> ContextualCompressionRetriever:
    """Return a retrieval + rerank pipeline scoped to a specific date."""
    reranker = CrossEncoderReranker(model=_get_reranker_model(), top_n=top_n)
    return ContextualCompressionRetriever(
        base_retriever=_ArxivDateRetriever(date=target_date),
        base_compressor=reranker,
    )


def _dedupe_chunks(docs: list[Document]) -> list[str]:
    """Map reranked chunks back to their full papers, deduped by arxiv id."""
    seen: set[str] = set()
    papers = []
    for doc in docs:
        paper_id = doc.metadata["arxiv_id"]
        if paper_id in seen:
            continue
        seen.add(paper_id)
        papers.append(format_arxiv_paper(
            title=doc.metadata["title"],
            url=doc.metadata["url"],
            abstract=doc.metadata["abstract"],
        ))
    return papers


def _recommend_query_based_by_date(target_date: str, query: str) -> list[str]:
    """Recommend a date's papers ranked by relevance to a query."""
    compression_retriever = _get_compression_retriever(target_date)
    retrieved_doc_splits = compression_retriever.invoke(query)
    return _dedupe_chunks(retrieved_doc_splits)[:MAX_RECOMMENDATIONS]


_bookmark_cache: dict[str, list[dict[str, Any]]] = {}


def _fetch_bookmarked_papers() -> list[dict[str, Any]]:
    """Fetch bookmarked papers, cached by combined urls-files content hash."""
    from arxiv_agent.tools import bookmarked_arxiv_papers

    files = bookmarked_arxiv_papers._get_bookmarked_urls_files()
    combined = "".join(f.read_text() for f in files)
    cache_key = hashlib.sha1(combined.encode()).hexdigest()
    if cache_key not in _bookmark_cache:
        _bookmark_cache.clear()
        _bookmark_cache[cache_key] = bookmarked_arxiv_papers._fetch_papers(with_abstract=True)
    return _bookmark_cache[cache_key]


def _recommend_personalized_by_date(target_date: str) -> list[str]:
    """Recommend a date's papers ranked by how many bookmarks surface them."""
    bookmarked_papers = _fetch_bookmarked_papers()
    if not bookmarked_papers:
        return []

    candidates: dict[str, int] = {}  # arxiv_id -> match count
    for paper in bookmarked_papers[:MAX_BOOKMARKS_PROCESSED]:
        query = format_arxiv_paper(
            title=paper["title"],
            url=paper["url"],
            abstract=paper["abstract"],
        )
        compression_retriever = _get_compression_retriever(target_date)
        retrieved_doc_splits = compression_retriever.invoke(query)
        for doc in retrieved_doc_splits:
            paper_id = doc.metadata["arxiv_id"]
            candidates[paper_id] = candidates.get(paper_id, 0) + 1

    papers_by_id = {row["arxiv_id"]: row for row in db.papers_by_date(target_date)}
    feed_order = {paper_id: index for index, paper_id in enumerate(papers_by_id)}
    ranked_ids = sorted(
        candidates,
        key=lambda paper_id: (-candidates[paper_id], feed_order.get(paper_id, 0)),
    )
    return [
        format_arxiv_paper(
            title=papers_by_id[paper_id]["title"],
            url=papers_by_id[paper_id]["url"],
            abstract=papers_by_id[paper_id]["abstract"],
        )
        for paper_id in ranked_ids[:MAX_RECOMMENDATIONS]
    ]


def _empty_date_message(target_date: str, most_recent_date: str | None) -> str:
    """Explain an empty arXiv corpus for a date, using universal (UTC) time."""
    target = datetime.strptime(target_date, "%Y-%m-%d")
    weekday = target.strftime("%A")
    if target.weekday() >= 5:
        explanation = (
            "arXiv announces papers Sunday–Thursday in America/New_York "
            "(Monday–Friday in UTC), so UTC weekend dates have no announcements. "
        )
    else:
        explanation = "arXiv announcements for this date have not been generated yet. "
    message = f"No papers for {target_date} (UTC, {weekday}). {explanation}"
    if most_recent_date:
        message += f"Last date with papers: {most_recent_date}."
    return message


@tool
def recommend_arxiv_papers_by_date(
    target_date: str,
    mode: Literal["query_based", "personalized"],
    query: str | None = None,
) -> str:
    """Recommend a date's arXiv papers based on either a query or the user's bookmarked papers."""
    _ensure_synced(target_date)

    if not db.date_has_chunks(target_date):
        return _empty_date_message(target_date, db.get_most_recent_date())

    if mode == "query_based":
        if query is None:
            return "Query must be provided for query-based recommendations."
        retrieved_docs = _recommend_query_based_by_date(target_date, query)
        if not retrieved_docs:
            return "No matching papers found for query."
        return "\n\n".join(retrieved_docs)
        
    if mode == "personalized":
        retrieved_docs = _recommend_personalized_by_date(target_date)
        if not retrieved_docs:
            return "No matching papers found for bookmarked papers."
        return "\n\n".join(retrieved_docs)
    
    return "Invalid mode. Please choose 'query_based' or 'personalized'."


@tool
def recommend_todays_arxiv_papers(mode: Literal["query_based", "personalized"], query: str | None = None) -> str:
    """Recommend today's arXiv papers based on either a query or the user's bookmarked papers."""
    try:
        return recommend_arxiv_papers_by_date.invoke(
            input={"target_date": date_cls.today().isoformat(), "mode": mode, "query": query}
        )
    except Exception as exc:  # pragma: no cover - defensive for server runs
        return f"Recommendation failed: {exc}"


# Test the tool function
if __name__ == "__main__":
    from pprint import pprint

    pprint(recommend_todays_arxiv_papers.invoke(input={"mode": "query_based", "query": "machine learning"}))
    pprint(recommend_todays_arxiv_papers.invoke(input={"mode": "personalized"}))
