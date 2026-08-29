from unittest.mock import Mock

from openai import APIConnectionError

from arxiv_agent.tools import recommend_arxiv_papers


def test_recommendation_hides_connection_details(monkeypatch, caplog) -> None:
    def raise_connection_error(_target_date: str) -> None:
        raise APIConnectionError(
            message="connection failed",
            request=Mock(),
        )

    monkeypatch.setattr(
        recommend_arxiv_papers,
        "_ensure_synced",
        raise_connection_error,
    )

    with caplog.at_level("ERROR", logger="arxiv_agent.tools.recommend_arxiv_papers"):
        result = recommend_arxiv_papers.recommend_arxiv_papers_by_date.invoke(
            {
                "target_date": "2026-08-29",
                "mode": "query_based",
                "query": "machine learning",
            }
        )

    assert result == recommend_arxiv_papers.EMBEDDING_UNAVAILABLE_MESSAGE
    assert "connection failed" not in result
    assert "Embedding service connection failed" in caplog.text
    assert "connection failed" in caplog.text
