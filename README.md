# arXiv Agent

A LangGraph-based AI agent that recommends arXiv papers for today or a specified date, based on bookmarked arXiv papers (personalized recommendations) or topics (query-based recommendations).

https://github.com/user-attachments/assets/e08d41aa-5a73-41e0-8637-c7ce12eb8a18

*Note: The demo video is played at 2× speed for brevity.*

## Quick Start

Visit https://arxiv-agent-chat.pages.dev?key=hostes, enter the access key, and start chatting with the agent.

## Running Locally

**Requirements:** [Git](https://git-scm.com/install/) and [Docker](https://docs.docker.com/get-started/get-docker/)

1. Open a terminal, clone the repository, and navigate to the repository directory.

```bash
git clone https://github.com/alxxtexxr/arxiv_agent.git && cd arxiv_agent
```

2. Create `.env.production` by copying `.env.production.example`, then set the API key and model configuration. Optionally, set the other configuration based on your preferences or needs.

```bash
cp .env.production.example .env.production
```

3. Start all services for the agent server.

```bash
docker compose up -d
```

4. The agent server should now be running at http://localhost:2024. Open this URL in a browser to verify that it is running. You should see:
```json
{
    "ok": "true"
}
```

5. To start chatting with the agent, set up the chat UI by following the `README.md` int the [arxiv-agent-chat-ui](https://github.com/alxxtexxr/arxiv-agent-chat-ui) repository, or more conveniently, use LangSmith Studio (https://smith.langchain.com/studio?baseUrl=http://localhost:2024).

## Development

**Requirements:** [Git](https://git-scm.com/install/), [Python](https://www.python.org/downloads/release/python-3147), [uv](https://docs.astral.sh/uv/getting-started/installation), and [PostgreSQL](https://www.postgresql.org/download)

1. Open a terminal, clone the repository, and navigate to the repository directory.

```bash
git clone https://github.com/alxxtexxr/arxiv_agent.git && cd arxiv_agent
```

2. Install the Python dependencies.

```bash
uv init && uv sync
```

3. Run the PostgreSQL server.

4. Create `.env` by copying `.env.example`, then set the API key and model configuration. Optionally, set the other configuration based on your preferences or needs.

```bash
cp .env.example .env
```

5. Create `src/arxiv_agent/data/bookmarked_arxiv_urls.txt` by copying `src/arxiv_agent/data/bookmarked_arxiv_urls.example.txt`. Then, update the file with the arXiv paper links for the bookmarks you want to use for personalized recommendations.

```bash
cp src/arxiv_agent/data/bookmarked_arxiv_urls.example.txt src/arxiv_agent/data/bookmarked_arxiv_urls.txt
```

6. Start the LangGraph agent server. You should be automatically redirected to LangSmith Studio, where you can start chatting with the agent.

```bash
uv run langgraph dev
``` 