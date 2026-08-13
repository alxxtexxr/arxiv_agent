"""Agent model definition."""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from arxiv_agent.tools import tools

load_dotenv() # Load environment variables from .env file

model = init_chat_model(
    model=f"{os.environ['MODEL_PROVIDER']}:{os.environ['MODEL_NAME']}",
    reasoning_effort=os.environ["MODEL_REASONING_EFFORT"],
)
model_with_tools = model.bind_tools(tools)
