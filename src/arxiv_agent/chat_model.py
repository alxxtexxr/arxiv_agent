"""Agent model definition."""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from arxiv_agent.tools import tools

load_dotenv() # Load environment variables from .env file

_chat_model_str = os.environ["CHAT_MODEL"]
# Support both "openai/gpt-..." (slash) and "openai:gpt-..." (colon) formats
if "/" in _chat_model_str and ":" not in _chat_model_str:
    _chat_model_str = _chat_model_str.replace("/", ":", 1)

chat_model = init_chat_model(
    model=_chat_model_str,
    reasoning_effort=os.environ["CHAT_MODEL_REASONING_EFFORT"],
)
chat_model_with_tools = chat_model.bind_tools(tools)
