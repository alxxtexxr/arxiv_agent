"""Blog tools and helpers."""

import os
import requests
from functools import lru_cache

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool

load_dotenv() # Load environment variables from .env file

def load_web_page(url: str, bs_kwargs: dict | None = None) -> list[Document]:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser", **(bs_kwargs or {}))
    return [Document(page_content=soup.get_text(), metadata={"source": url})]

filepath = os.path.join(os.path.dirname(__file__), "..", "data", "saved_blog_links.txt")
with open(filepath) as f:
    links = [line.strip() for line in f if line.strip()]
    
docs = [load_web_page(link) for link in links]
docs_list = [doc for sublist in docs for doc in sublist] # Flatten the list of lists

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
doc_splits = text_splitter.split_documents(docs_list)

@lru_cache(maxsize=1)
def _get_retriever():
    vectorstore = InMemoryVectorStore.from_documents(doc_splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()

@tool
def search_blog_posts(query: str) -> str:
    """Search for blog posts based on a query."""
    retriever = _get_retriever()
    retrieved_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])

# Test the tool
if __name__ == "__main__":
    print(search_blog_posts.invoke({"query": "types of reward hacking"}))