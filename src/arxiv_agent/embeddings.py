"""Embedding model factory supporting multiple strategies."""

import os
from typing import List

import requests
from anyio.functools import lru_cache
from langchain_core.embeddings import Embeddings
from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Return an embeddings instance based on EMBEDDING_STRATEGY.

    Strategies:
    - ``openai``: uses ``OpenAIEmbeddings`` with ``text-embedding-3-small``.
    - ``tei``: uses ``OpenAIEmbeddings`` against TEI's OpenAI-compatible
      ``/v1/embeddings`` endpoint (``TEI_EMBEDDING_URL`` + ``/v1``).
    - ``hf`` / ``huggingface``: uses local ``HuggingFaceEmbeddings`` with
      ``EMBEDDING_MODEL``.
    """
    strategy = os.environ["EMBEDDING_STRATEGY"].lower()

    if strategy == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    if strategy == "tei":
        from langchain_openai import OpenAIEmbeddings

        base_url = os.environ.get("TEI_EMBEDDING_URL") or os.environ.get("TEI_EMBEDDINNG_URL")
        if not base_url:
            raise ValueError("TEI_EMBEDDING_URL (or TEI_EMBEDDINNG_URL) must be set for EMBEDDING_STRATEGY=tei")
        return OpenAIEmbeddings(
            model=os.environ["EMBEDDING_MODEL"],
            base_url=base_url.rstrip("/") + "/v1",
            api_key="unused",
            check_embedding_ctx_length=False,
            chunk_size=32,
            max_retries=2,
        )

    if strategy in ("hf", "huggingface"):
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=os.environ["EMBEDDING_MODEL"],
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(
        f"Unknown EMBEDDING_STRATEGY '{strategy}'. Expected 'openai' | 'tei' | 'hf' | 'huggingface'."
    )


@lru_cache(maxsize=1)
def _get_reranker_model():
    """Return the shared reranker model instance."""
    strategy = os.environ["RERANKER_STRATEGY"].lower()
    
    if strategy == "tei":
        base_url = os.environ.get("TEI_RERANKER_URL")
        if not base_url:
            raise ValueError("TEI_RERANKER_URL must be set for RERANKER_STRATEGY=tei")
        endpoint = base_url.rstrip("/") + "/rerank"
        
        class TeiCrossEncoder(BaseCrossEncoder):
            def __init__(self, endpoint_url: str):
                self.endpoint_url = endpoint_url

            def score(self, pairs: List[List[str]]) -> List[float]:
                if not pairs:
                    return []
                
                query = pairs[0][0]
                texts = [text for _, text in pairs]
                
                response = requests.post(
                    self.endpoint_url,
                    json={"query": query, "texts": texts, "raw_scores": False},
                    timeout=60,
                )
                response.raise_for_status()
                results = response.json()
                
                # TEI returns:
                # [{"index": 1, "score": 0.94}, {"index": 0, "score": 0.12}]
                #
                # BaseCrossEncoder expects scores in the same order
                # as text_pairs.
                return [
                    next(result["score"] for result in results if result["index"] == i)
                    for i in range(len(texts))
                ]

        return TeiCrossEncoder(endpoint)
    
    if strategy in ("hf", "huggingface"):
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    
        return HuggingFaceCrossEncoder(model_name=os.environ["RERANKER_MODEL"])

    raise ValueError(
        f"Unknown RERANKER_STRATEGY: {strategy}. Expected 'tei' | 'hf' | 'huggingface'."
    )