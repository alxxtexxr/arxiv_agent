"""Reranker model factory."""

import os
from typing import List

import requests
from anyio.functools import lru_cache
from dotenv import load_dotenv
from langchain_classic.retrievers.document_compressors.cross_encoder import (
    BaseCrossEncoder,
)

load_dotenv()

ENABLE_RERANKER_CACHE = os.getenv("ENABLE_RERANKER_CACHE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _get_reranker_model_uncached():
    """Return the shared reranker model instance."""
    strategy = os.environ["RERANKER_INTEGRATION"].lower()
    
    if strategy == "tei":
        base_url = os.environ.get("TEI_RERANKER_URL")
        if not base_url:
            raise ValueError("TEI_RERANKER_URL must be set for RERANKER_INTEGRATION=tei")
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
        f"Unknown RERANKER_INTEGRATION: '{strategy}'. Expected one of: 'tei' | 'hf' | 'huggingface'."
    )


_get_reranker_model = (
    lru_cache(maxsize=1)(_get_reranker_model_uncached)
    if ENABLE_RERANKER_CACHE
    else _get_reranker_model_uncached
)

# Keep backward-compatible alias (some modules import _get_reranker_model)
get_reranker_model = _get_reranker_model
