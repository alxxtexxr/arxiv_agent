"""Agent model definition."""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from arxiv_agent.tools import tools

load_dotenv() # Load environment variables from .env file

_chat_model_str = os.environ["CHAT_MODEL"]
if len(_chat_model_str.split(":")) != 2:
    raise ValueError(
        f"Invalid CHAT_MODEL: '{_chat_model_str}'. Expected format: 'model_provider:model_name'."
    )

if "mistral" in _chat_model_str:
    from langchain_mistralai import ChatMistralAI
    
    chat_model = ChatMistralAI(model=_chat_model_str.split(":")[1])
else:
    chat_model = init_chat_model(
        model=_chat_model_str,
        reasoning_effort=os.environ["CHAT_MODEL_REASONING_EFFORT"],
    )

chat_model_with_tools = chat_model.bind_tools(tools)
