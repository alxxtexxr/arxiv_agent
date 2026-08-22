# arXiv Agent

A LangGraph AI agent that recommends arXiv papers for today or a specified date, based on bookmarked arXiv papers (personalized recommendations) or topics (query-based recommendations).

https://github.com/user-attachments/assets/e08d41aa-5a73-41e0-8637-c7ce12eb8a18

*Note: The demo video runs at 2× speed for brevity.*

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

4. Set your PostgreSQL database URI and [OpenAI API key](https://developers.openai.com/api/docs/quickstart#create-and-export-an-api-key) in `.env`. Optionally, update the OpenAI model settings.

```dotenv
ARXIV_AGENT_POSTGRES_URI="postgresql://{username}:{password}@localhost:5432/arxiv_agent"

OPENAI_API_KEY="sk-..."
MODEL_PROVIDER="openai"
MODEL_NAME="gpt-5.6-luna"
MODEL_REASONING_EFFORT="max"
```

5. Create `src/arxiv_agent/data/bookmarked_arxiv_links.txt` by copying `src/arxiv_agent/data/bookmarked_arxiv_links.example.txt`. Then, update the file with the arXiv paper links for the bookmarks you want to use for personalized recommendations.

```bash
cp src/arxiv_agent/data/bookmarked_arxiv_links.example.txt src/arxiv_agent/data/bookmarked_arxiv_links.txt
```

6. Start the LangGraph server. You should be automatically redirected to LangSmith Studio, where you can start chatting with the agent.

```bash
uv run langgraph dev
```

<!-- ## TODO

- [ ] Add Docker support for the application. -->
