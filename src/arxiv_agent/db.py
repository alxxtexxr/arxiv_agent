"""PostgreSQL + pgvector storage layer for arXiv papers.

Hand-rolled SQL (no ORM/vectorstore wrapper) with a normalized schema: paper
metadata lives once in ``arxiv_papers``, while chunks and their configured
embeddings live in ``arxiv_paper_chunks``. Idempotent upserts are backed by
unique constraints on the natural keys (arxiv_id / arxiv_id + chunk_idx), a
btree index enables date-based retrieval, and vector searches are filtered by
embedding dimensionality so stale vectors from older embedding models cannot
crash pgvector comparisons.
"""

import os

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS arxiv_papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    arxiv_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    abstract TEXT NOT NULL,
    published_date DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS arxiv_paper_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    arxiv_id TEXT NOT NULL REFERENCES arxiv_papers (arxiv_id) ON DELETE CASCADE,
    chunk_idx INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (arxiv_id, chunk_idx)
);

CREATE TABLE IF NOT EXISTS arxiv_sync_meta (
    published_date DATE PRIMARY KEY,
    embedding_version TEXT NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_arxiv_papers_date
    ON arxiv_papers (published_date);

"""

UPSERT_PAPER_SQL = """
INSERT INTO arxiv_papers (arxiv_id, title, url, abstract, published_date)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (arxiv_id) DO UPDATE SET
    title = EXCLUDED.title,
    url = EXCLUDED.url,
    abstract = EXCLUDED.abstract,
    published_date = EXCLUDED.published_date
"""

DELETE_CHUNKS_SQL = """
DELETE FROM arxiv_paper_chunks
WHERE arxiv_id IN (SELECT arxiv_id FROM arxiv_papers WHERE published_date = %s)
"""

UPSERT_CHUNK_SQL = """
INSERT INTO arxiv_paper_chunks (arxiv_id, chunk_idx, content, embedding)
VALUES (%s, %s, %s, %s)
"""

UPSERT_SYNC_META_SQL = """
INSERT INTO arxiv_sync_meta (published_date, embedding_version)
VALUES (%s, %s)
ON CONFLICT (published_date) DO UPDATE SET
    embedding_version = EXCLUDED.embedding_version,
    synced_at = now()
"""

GET_SYNC_VERSION_SQL = """
SELECT embedding_version FROM arxiv_sync_meta WHERE published_date = %s
"""

MOST_RECENT_DATE_SQL = """
SELECT published_date FROM arxiv_sync_meta
ORDER BY published_date DESC LIMIT 1
"""

SEARCH_SQL = """
SELECT c.arxiv_id, c.chunk_idx, p.title, p.url, p.abstract, c.content,
       1 - (c.embedding <=> %s) AS score
FROM arxiv_paper_chunks c
JOIN arxiv_papers p ON p.arxiv_id = c.arxiv_id
WHERE p.published_date = %s
  AND vector_dims(c.embedding) = %s
ORDER BY c.embedding <=> %s
LIMIT %s
"""

DATE_HAS_PAPERS_SQL = """
SELECT EXISTS (SELECT 1 FROM arxiv_papers WHERE published_date = %s)
"""

DATE_HAS_CHUNKS_SQL = """
SELECT EXISTS (
    SELECT 1 FROM arxiv_paper_chunks c
    JOIN arxiv_papers p ON p.arxiv_id = c.arxiv_id
    WHERE p.published_date = %s
)
"""

CLEANUP_ORPHAN_PAPERS_SQL = """
DELETE FROM arxiv_papers
WHERE published_date = %s
  AND NOT EXISTS (
      SELECT 1 FROM arxiv_paper_chunks c WHERE c.arxiv_id = arxiv_papers.arxiv_id
  )
"""

PAPERS_BY_DATE_SQL = """
SELECT arxiv_id, title, url, abstract, created_at AS first_seen
FROM arxiv_papers
WHERE published_date = %s
ORDER BY created_at
"""


def _connect() -> psycopg.Connection:
    """Open an autocommit connection; pgvector types are registered per connection."""
    connection = psycopg.connect(os.environ["VECTOR_DATABASE_URI"], autocommit=True)
    register_vector(connection)
    return connection


def init_schema() -> None:
    """Create the extension, tables, and indexes if they do not exist.

    Uses an unregistered connection: the schema SQL creates the ``vector``
    type itself, so ``register_vector`` (which requires the type to exist)
    cannot run before it.
    """
    with psycopg.connect(os.environ["VECTOR_DATABASE_URI"], autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)
            # A global HNSW index is unsafe for an unconstrained VECTOR column when
            # embedding models can change dimensions; date-scoped scans are small.
            cursor.execute("DROP INDEX IF EXISTS idx_arxiv_paper_chunks_embedding")
            # Migrate legacy `link` column to `url` if needed (no-op for fresh installs)
            cursor.execute("""
                DO $$ BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='arxiv_papers' AND column_name='link'
                    ) THEN
                        ALTER TABLE arxiv_papers RENAME COLUMN link TO url;
                    END IF;
                END $$;
            """)
            # Migrate legacy fixed-dimension vector to unconstrained for strategy flexibility
            cursor.execute("""
                DO $$ BEGIN
                    BEGIN
                        ALTER TABLE arxiv_paper_chunks ALTER COLUMN embedding TYPE VECTOR;
                    EXCEPTION WHEN OTHERS THEN NULL;
                    END;
                END $$;
            """)


def date_has_papers(published_date: str) -> bool:
    """Return whether any papers are stored for the given date."""
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(DATE_HAS_PAPERS_SQL, (published_date,))
        return cursor.fetchone()[0]


def date_has_chunks(published_date: str) -> bool:
    """Return whether any chunks are stored for the given date."""
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(DATE_HAS_CHUNKS_SQL, (published_date,))
        return cursor.fetchone()[0]


def get_sync_version(published_date: str) -> str | None:
    """Return the stored embedding version for a date, or None if unsynced."""
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(GET_SYNC_VERSION_SQL, (published_date,))
        row = cursor.fetchone()
        return row[0] if row else None


def get_most_recent_date() -> str | None:
    """Return the most recent date with stored papers, or None."""
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(MOST_RECENT_DATE_SQL)
        row = cursor.fetchone()
        return row[0].isoformat() if row and row[0] else None


def sync_date(published_date: str, rows: list[dict], embedding_version: str) -> None:
    """Atomically sync a date's papers, chunks, and sync metadata.

    Upserts paper metadata, replaces the date's chunks, and records the
    embedding version that produced them. Chunks are deleted and re-inserted
    (rather than upserted) so that a change in the chunking configuration
    cannot leave stale chunks with old boundaries behind. All phases run in a
    single transaction.
    """
    paper_params = [
        (
            row["arxiv_id"],
            row["title"],
            row["url"],
            row["abstract"],
            published_date,
        )
        for row in rows
    ]
    chunk_params = [
        (
            row["arxiv_id"],
            row["chunk_idx"],
            row["content"],
            Vector(row["embedding"]),
        )
        for row in rows
    ]
    with _connect() as connection, connection.transaction():
        with connection.cursor() as cursor:
            cursor.executemany(UPSERT_PAPER_SQL, paper_params)
            cursor.execute(DELETE_CHUNKS_SQL, (published_date,))
            cursor.executemany(UPSERT_CHUNK_SQL, chunk_params)
            cursor.execute(UPSERT_SYNC_META_SQL, (published_date, embedding_version))
            cursor.execute(CLEANUP_ORPHAN_PAPERS_SQL, (published_date,))


def search_by_date(published_date: str, query_vector: list[float], k: int) -> list[dict]:
    """Return the top-k chunks for a date, ordered by cosine similarity."""
    vector = Vector(query_vector)
    with _connect() as connection:
        with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
            cursor.execute(SEARCH_SQL, (vector, published_date, len(query_vector), vector, k))
            return cursor.fetchall()


def papers_by_date(published_date: str) -> list[dict]:
    """Return distinct paper metadata for a date, in first-seen order."""
    with _connect() as connection:
        with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
            cursor.execute(PAPERS_BY_DATE_SQL, (published_date,))
            return cursor.fetchall()
