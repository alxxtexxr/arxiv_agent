"""Agent graph definition."""

from langgraph.graph import StateGraph

from agent.nodes import call_model, call_tool, call_tool_or_stop
from agent.state import Context, State

graph = (
    StateGraph(State, context_schema=Context)
    
    # Add nodes to the graph
    .add_node("call_model", call_model)
    .add_node("call_tool", call_tool)
    
    # Add edges to define the flow of the graph
    .add_edge("__start__", "call_model")
    .add_conditional_edges("call_model", call_tool_or_stop, ["call_tool", "__end__"])
    .add_edge("call_tool", "call_model")
    
    # Compile the graph
    .compile(
        # name="My Agent Graph",
    )
)
