"""Agent state and context definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """
    my_configurable_param: str

@dataclass
class State(TypedDict):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """
    messages: Annotated[list[AnyMessage], add_messages]
    num_model_calls: int
    saved_arxiv_papers: list[dict[str, Any]]
