"""PostgreSQL + pgvector storage layer for arXiv papers.

Hand-rolled SQL (no ORM/vectorstore wrapper): an explicit schema with a
unique constraint per (arxiv_id, chunk_idx) for idempotent upserts, a btree
index on the publication date for date-based retrieval, and an HNSW index on
the embeddings for approximate cosine similarity search.
"""

import os

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS arxiv_papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    arxiv_id TEXT NOT NULL,
    chunk_idx INT NOT NULL,
    title TEXT NOT NULL,
    link TEXT NOT NULL,
    abstract TEXT NOT NULL,
    published_date DATE NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (arxiv_id, chunk_idx)
);

CREATE INDEX IF NOT EXISTS idx_arxiv_papers_date
    ON arxiv_papers (published_date);

CREATE INDEX IF NOT EXISTS idx_arxiv_papers_embedding
    ON arxiv_papers USING hnsw (embedding vector_cosine_ops);
"""

UPSERT_SQL = """
INSERT INTO arxiv_papers
    (arxiv_id, chunk_idx, title, link, abstract, published_date, content, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (arxiv_id, chunk_idx) DO UPDATE SET
    title = EXCLUDED.title,
    link = EXCLUDED.link,
    abstract = EXCLUDED.abstract,
    content = EXCLUDED.content,
    embedding = EXCLUDED.embedding
"""

# '<=>' computes cosine distance (smaller is better)
# cosine distance = 1 - cosine similarity
# score = 1 - cosine distance = cosine similarity
SEARCH_SQL = """
SELECT arxiv_id, chunk_idx, title, link, abstract, content,
       1 - (embedding <=> %s) AS score
FROM arxiv_papers
WHERE published_date = %s
ORDER BY embedding <=> %s
LIMIT %s
"""

DATE_HAS_PAPERS_SQL = """
SELECT EXISTS (SELECT 1 FROM arxiv_papers WHERE published_date = %s)
"""

PAPERS_BY_DATE_SQL = """
SELECT arxiv_id, title, link, abstract, min(created_at) AS first_seen
FROM arxiv_papers
WHERE published_date = %s
GROUP BY arxiv_id, title, link, abstract
ORDER BY first_seen
"""


def _connect() -> psycopg.Connection:
    """Open an autocommit connection; pgvector types are registered per connection."""
    connection = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    register_vector(connection)
    return connection


def init_schema() -> None:
    """Create the extension, table, and indexes if they do not exist.

    Uses an unregistered connection: the schema SQL creates the ``vector``
    type itself, so ``register_vector`` (which requires the type to exist)
    cannot run before it.
    """
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)


def date_has_papers(published_date: str) -> bool:
    """Return whether any papers are stored for the given date."""
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(DATE_HAS_PAPERS_SQL, (published_date,))
        return cursor.fetchone()[0]


def upsert_papers(published_date: str, rows: list[dict]) -> None:
    """Idempotently insert or update paper chunks for a date."""
    params = [
        (
            row["arxiv_id"],
            row["chunk_idx"],
            row["title"],
            row["link"],
            row["abstract"],
            published_date,
            row["content"],
            Vector(row["embedding"]),
        )
        for row in rows
    ]
    with _connect() as connection, connection.cursor() as cursor:
        cursor.executemany(UPSERT_SQL, params)


def search_by_date(published_date: str, query_vector: list[float], k: int) -> list[dict]:
    """Return the top-k chunks for a date, ordered by cosine similarity."""
    vector = Vector(query_vector)
    with _connect() as connection:
        with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
            cursor.execute(SEARCH_SQL, (vector, published_date, vector, k))
            return cursor.fetchall()


def papers_by_date(published_date: str) -> list[dict]:
    """Return distinct paper metadata for a date, in first-seen order."""
    with _connect() as connection:
        with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
            cursor.execute(PAPERS_BY_DATE_SQL, (published_date,))
            return cursor.fetchall()
