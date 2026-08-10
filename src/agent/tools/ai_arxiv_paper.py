from typing import Literal
from builtins import sorted

import feedparser
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from anyio.functools import lru_cache
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.tools import tool

from agent.utils import format_arxiv_paper

load_dotenv() # Load environment variables from .env file

AI_ARXIV_RSS_URL = "http://rss.arxiv.org/rss/cs.AI"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

feed = feedparser.parse(AI_ARXIV_RSS_URL)
paper_docs = [
    format_arxiv_paper(
        title=entry.title, 
        link=entry.link, 
        abstract=entry.summary.split("Abstract:", 1)[-1].strip(),
    ) for entry in feed.entries
]

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
    """Get a compression retriever for today's arXiv AI papers."""
    
    # Define the embedding model and retriever
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    retriever = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=embedding_model,
    ).as_retriever()
    
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

def _recommend_query_based(query: str) -> list[str]:
    """Recommend today's arXiv AI papers based on a query."""
    compression_retriever = _get_compression_retriever()
    retrieved_doc_splits = compression_retriever.invoke(query)
    retrieved_doc_indices = sorted({doc.metadata["paper_index"] for doc in retrieved_doc_splits})
    retrieved_docs = [paper_docs[i] for i in retrieved_doc_indices]
    return retrieved_docs

def _recommend_personalized() -> list[str]:
    """Recommend today's arXiv AI papers based on personalized preferences."""
    # Placeholder for personalized recommendation logic
    # This could involve user profiles, past interactions, etc.
    return []

@tool
def recommend_todays_arxiv_ai_papers(mode: Literal["query_based", "personalized"], query: str | None = None) -> str:
    """Recommend today's arXiv AI papers based on the specified mode."""
    retrieved_docs = []
    if mode == "query_based":
        assert query is not None, "Query must be provided for query-based recommendations."
        retrieved_docs = _recommend_query_based(query)
    elif mode == "personalized":
        retrieved_docs = _recommend_personalized()
    else:
        raise ValueError("Invalid mode. Please choose 'query_based' or 'personalized'.")
    return "\n\n".join(retrieved_docs)

# Test the tool function
if __name__ == "__main__":
    from pprint import pprint
    pprint(recommend_todays_arxiv_ai_papers.invoke(input={"mode": "query_based", "query": "machine learning"}))