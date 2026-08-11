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

from agent import db
from agent.utils import format_arxiv_paper

load_dotenv()

ARXIV_CATEGORY = "cs.AI"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_RECOMMENDATIONS = 5
MAX_BOOKMARKS_PROCESSED = 10

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
    feed = feedparser.parse(f"http://rss.arxiv.org/rss/{ARXIV_CATEGORY}")
    return [
        {
            "arxiv_id": _normalize_arxiv_id(entry.link),
            "title": entry.title,
            "link": entry.link,
            "abstract": entry.summary.split("Abstract:", 1)[-1].strip(),
        }
        for entry in feed.entries
    ]


def _fetch_api_entries_for_date(target_date: str) -> list[dict[str, str]]:
    """Fetch papers submitted on a specific date via the arXiv API."""
    compact_date = target_date.replace("-", "")
    query = (
        f"cat:{ARXIV_CATEGORY} AND "
        f"submittedDate:[{compact_date}0000 TO {compact_date}2359]"
    )
    search = arxiv_api.Search(
        query=query,
        max_results=2000,
        sort_by=arxiv_api.SortCriterion.SubmittedDate,
        sort_order=arxiv_api.SortOrder.Ascending,
    )
    return [
        {
            "arxiv_id": _normalize_arxiv_id(result.entry_id),
            "title": result.title,
            "link": f"https://arxiv.org/abs/{result.entry_id}",
            "abstract": result.summary,
        }
        for result in arxiv_api.Client().results(search)
    ]


def _sync_date(target_date: str) -> int:
    """Fetch, chunk, embed, and upsert a date's papers. Returns chunk count."""
    if target_date == date_cls.today().isoformat():
        entries = _fetch_feed_entries()
    else:
        entries = _fetch_api_entries_for_date(target_date)
    if not entries:
        return 0

    paper_docs = [
        format_arxiv_paper(title=e["title"], link=e["link"], abstract=e["abstract"])
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
                "link": entry["link"],
                "abstract": entry["abstract"],
                "content": chunk_text,
                "embedding": flat_embeddings[offset],
            })
            offset += 1

    db.upsert_papers(target_date, rows)
    return len(rows)


def _ensure_synced(target_date: str) -> None:
    """Ensure a date's papers are stored, syncing once per date per process."""
    _ensure_db_ready()
    if target_date in _synced_dates:
        return
    if not db.date_has_papers(target_date):
        _sync_date(target_date)
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
                    "link": row["link"],
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
            link=doc.metadata["link"],
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
    """Fetch bookmarked papers, cached by links-file content hash."""
    from agent.tools import bookmarked_arxiv_paper

    links_file = bookmarked_arxiv_paper.BOOKMARKED_ARXIV_LINKS_FILE
    cache_key = hashlib.sha1(links_file.read_text().encode()).hexdigest()
    if cache_key not in _bookmark_cache:
        _bookmark_cache.clear()
        _bookmark_cache[cache_key] = bookmarked_arxiv_paper._fetch_data()
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
            link=paper["link"],
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
            link=papers_by_id[paper_id]["link"],
            abstract=papers_by_id[paper_id]["abstract"],
        )
        for paper_id in ranked_ids[:MAX_RECOMMENDATIONS]
    ]


def _recommend_by_date(
    target_date: str,
    mode: Literal["query_based", "personalized"],
    query: str | None = None,
) -> list[str]:
    """Recommend papers published on a date. Internal core for the tools."""
    _ensure_synced(target_date)
    if mode == "query_based":
        assert query is not None, "Query must be provided for query-based recommendations."
        return _recommend_query_based_by_date(target_date, query)
    if mode == "personalized":
        return _recommend_personalized_by_date(target_date)
    raise ValueError("Invalid mode. Please choose 'query_based' or 'personalized'.")


@tool
def recommend_todays_arxiv_papers(mode: Literal["query_based", "personalized"], query: str | None = None) -> str:
    """Recommend today's arXiv papers based on either a query or the user's bookmarked papers."""
    try:
        retrieved_docs = _recommend_by_date(date_cls.today().isoformat(), mode, query)
    except Exception as exc:  # pragma: no cover - defensive for server runs
        return f"Recommendation failed: {exc}"

    if not retrieved_docs:
        if mode == "query_based":
            return "No matching papers found for today."
        return "No bookmarked papers found to base personalized recommendations on."
    return "\n\n".join(retrieved_docs)


# Test the tool function
if __name__ == "__main__":
    from pprint import pprint

    pprint(recommend_todays_arxiv_papers.invoke(input={"mode": "query_based", "query": "machine learning"}))
    pprint(recommend_todays_arxiv_papers.invoke(input={"mode": "personalized"}))
