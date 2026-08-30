"""Embedding model factory supporting multiple strategies."""

import os

from anyio.functools import lru_cache
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _get_embedding_model_uncached() -> Embeddings:
    """Return an embeddings instance based on EMBEDDING_INTEGRATION.

    Strategies:
    - ``openai``: uses ``OpenAIEmbeddings`` with ``text-embedding-3-small``.
    - ``tei``: uses ``OpenAIEmbeddings`` against TEI's OpenAI-compatible
      ``/v1/embeddings`` endpoint (``TEI_EMBEDDING_URL`` + ``/v1``).
    - ``hf`` / ``huggingface``: uses local ``HuggingFaceEmbeddings`` with
      ``EMBEDDING_MODEL``.
    """
    strategy = os.environ["EMBEDDING_INTEGRATION"].lower()

    if strategy == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model_name=os.environ["EMBEDDING_MODEL"])

    if strategy == "fastembed":
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=os.environ["EMBEDDING_MODEL"])

    if strategy == "tei":
        from langchain_openai import OpenAIEmbeddings

        base_url = os.environ.get("TEI_EMBEDDING_URL") or os.environ.get("TEI_EMBEDDINNG_URL")
        if not base_url:
            raise ValueError("TEI_EMBEDDING_URL (or TEI_EMBEDDINNG_URL) must be set for EMBEDDING_INTEGRATION=tei")
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
        f"Unknown EMBEDDING_INTEGRATION: '{strategy}'. Expected one of: 'openai' | 'tei' | 'hf' | 'huggingface'."
    )


get_embedding_model = (
    lru_cache(maxsize=1)(_get_embedding_model_uncached)
    if ENABLE_EMBEDDING_CACHE
    else _get_embedding_model_uncached
)