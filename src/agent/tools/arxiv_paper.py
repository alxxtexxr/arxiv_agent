"""arXiv paper recommendation tool.

Fetches today's arXiv feed, embeds the papers with a local embedding model
(BAAI/bge-m3), stores the vectors in a FAISS index persisted to disk, and
recommends papers via retrieval plus cross-encoder reranking.
"""

import hashlib
import os
from builtins import sorted
from typing import Literal

import feedparser
from anyio.functools import lru_cache
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS

# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.utils import format_arxiv_paper

load_dotenv() # Load environment variables from .env file

ARXIV_CATEGORY = "cs.AI"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

try:
    feed = feedparser.parse(f"http://rss.arxiv.org/rss/{ARXIV_CATEGORY}")
    paper_docs = [
        format_arxiv_paper(
            title=entry.title, 
            link=entry.link, 
            abstract=entry.summary.split("Abstract:", 1)[-1].strip(),
        ) for entry in feed.entries
    ]
except Exception as e:
    print(f"Error fetching or parsing the arXiv feed: {e}")
    paper_docs = []

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
doc_splits = text_splitter.create_documents(
    paper_docs,
    metadatas=[{"paper_index": i} for i in range(len(paper_docs))],
)

@lru_cache(maxsize=1)
def _get_compression_retriever(top_n=5) -> ContextualCompressionRetriever:
    """Get a compression retriever for today's arXiv papers."""
    # Define the embedding model and retriever
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    
    # Persist the FAISS index on disk, keyed by the feed content hash, so that
    # process restarts/reloads skip re-embedding the full feed (~60s) and load
    # the index in <1s. The index is rebuilt automatically when the feed changes.
    feed_hash = hashlib.sha1("\n".join(paper_docs).encode()).hexdigest()[:12]
    store_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", 
        f"arxiv_{ARXIV_CATEGORY.replace('.', '_').lower()}_faiss_{feed_hash}",
    )
    
    if os.path.isdir(store_dir):
        # allow_dangerous_deserialization is required because the FAISS index
        # pickles its document store. It is safe here because the index is only
        # ever loaded from a path this module wrote itself.
        store = FAISS.load_local(
            store_dir,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
    else:
        store = FAISS.from_documents(doc_splits, embedding_model)
        store.save_local(store_dir)
    
    retriever = store.as_retriever()
    
    # Define the reranker model and compression retriever
    reranker_model = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL_NAME,
    )
    reranker = CrossEncoderReranker(model=reranker_model, top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever, 
        base_compressor=reranker,
    )
    
    return compression_retriever

def _recommend_data_query_based(query: str) -> list[str]:
    """Recommend today's arXiv papers based on a query."""
    compression_retriever = _get_compression_retriever()
    retrieved_doc_splits = compression_retriever.invoke(query)
    retrieved_doc_indices = sorted({doc.metadata["paper_index"] for doc in retrieved_doc_splits})
    retrieved_docs = [paper_docs[i] for i in retrieved_doc_indices]
    return retrieved_docs

def _recommend_data_personalized() -> list[str]:
    """Recommend today's arXiv papers based on personalized criteria."""
    # Placeholder for personalized recommendation logic
    # This could involve user profiles, past interactions, etc.
    return []

@tool
def recommend_todays_arxiv_papers(mode: Literal["query_based", "personalized"], query: str | None = None) -> str:
    """Recommend today's arXiv papers based on the specified mode."""
    retrieved_docs = []
    if mode == "query_based":
        assert query is not None, "Query must be provided for query-based recommendations."
        retrieved_docs = _recommend_data_query_based(query)
    elif mode == "personalized":
        retrieved_docs = _recommend_data_personalized()
    else:
        raise ValueError("Invalid mode. Please choose 'query_based' or 'personalized'.")
    return "\n\n".join(retrieved_docs)

# Test the tool function
if __name__ == "__main__":
    from pprint import pprint
    pprint(recommend_todays_arxiv_papers.invoke(input={"mode": "query_based", "query": "machine learning"}))