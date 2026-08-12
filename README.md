# arXiv Agent

A LangGraph agent that recommends arXiv papers for today or a specified date, based on bookmarked arXiv papers (personalized recommendations) or topics (query-based recommendations).

## Requirements

- [Python](https://www.python.org/downloads/release/python-3147)
- [uv](https://docs.astral.sh/uv/getting-started/installation)
- [PostgreSQL](https://www.postgresql.org/download)

## Getting Started

1. Install the Python dependencies using uv.

```bash
uv init
uv sync
```

2. Run the PostgreSQL server.

3. Create `.env` by copying `.env.example`.

```bash
cp .env.example .env
```

4. Set your LangSmith API key, OpenAI API key, and PosgreSQL database URL. Follow [these instructions](https://docs.langchain.com/langsmith/create-account-api-key) to get the LangSmith API key, and [these instructions](https://developers.openai.com/api/docs/quickstart#create-and-export-an-api-key) to get the OpenAI API key. Optionally, update the OpenAI model settings.

```dotenv
# .env

LANGSMITH_API_KEY="lsv2_..."
OPENAI_API_KEY="sk-..."
DATABASE_URL="postgresql://{username}:{password}@localhost:5432/arxiv_agent"

MODEL_PROVIDER="openai"
MODEL_NAME="gpt-5.6-luna"
MODEL_REASONING_EFFORT="none"
```

5. Create `src/agent/data/bookmarked_arxiv_links.txt` by copying `src/agent/data/bookmarked_arxiv_links.example.txt`. Then, update the file with the arXiv paper links for the bookmarks you want to use for personalized recommendations.

```bash
cp src/agent/data/bookmarked_arxiv_links.example.txt src/agent/data/bookmarked_arxiv_links.txt
```

6. Start the LangGraph server. You should be automatically redirected to LangSmith Studio, where you can start chatting with the agent.

```bash
uv run langgraph dev
```

## TODO

- [ ] Add Docker support for the application.