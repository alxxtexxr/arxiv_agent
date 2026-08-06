from typing import Literal
from builtins import sorted

import feedparser
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from anyio.functools import lru_cache
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool

load_dotenv() # Load environment variables from .env file

rss_url = "http://rss.arxiv.org/rss/cs.AI"
feed = feedparser.parse(rss_url)

ARXIV_PAPER_DOC = "Title: {title}\nLink: {link}\nAbstract: {summary}"
paper_docs = [
    ARXIV_PAPER_DOC.format(
        title=entry.title, 
        link=entry.link, 
        summary=entry.summary.split("Abstract:", 1)[-1].strip(),
    ) for entry in feed.entries
]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=50,
)
doc_splits = text_splitter.create_documents(
    paper_docs,
    metadatas=[{"paper_index": i} for i in range(len(paper_docs))],
)

@lru_cache(maxsize=1)
def _get_retriever():
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(),
    )
    return vectorstore.as_retriever()

@tool
def recommend_todays_arxiv_ai_papers(mode: Literal["query_based", "personalized"], query: str | None = None) -> str:
    """Recommend today's arXiv AI papers based on the specified mode."""
    assert mode in ["query_based", "personalized"], "Mode must be either 'query_based' or 'personalized'."
    
    if mode == "query_based":
        assert query is not None, "Query must be provided for query-based recommendations."
    elif mode == "personalized":
        return "Personalized recommendations are not implemented yet. Please use 'query_based' mode with a query."
    
    retriever = _get_retriever()
    retrieved_docs = retriever.invoke(query)
    paper_indices = sorted({doc.metadata["paper_index"] for doc in retrieved_docs})
    
    return "\n\n".join(paper_docs[i] for i in paper_indices)

if __name__ == "__main__":
    from pprint import pprint
    pprint(recommend_todays_arxiv_ai_papers.invoke(input={"mode": "query_based", "query": "machine learning"}))