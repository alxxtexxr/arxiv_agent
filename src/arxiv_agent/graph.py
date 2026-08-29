"""Agent graph definition."""

from langgraph.graph import MessagesState, StateGraph

from arxiv_agent.nodes import call_chat_model, call_tool, call_tool_or_stop

graph = (
    StateGraph(MessagesState)
    
    # Add nodes to the graph
    .add_node("call_chat_model", call_chat_model)
    .add_node("call_tool", call_tool)
    
    # Add edges to define the flow of the graph
    .add_edge("__start__", "call_chat_model")
    .add_conditional_edges("call_chat_model", call_tool_or_stop, ["call_tool", "__end__"])
    .add_edge("call_tool", "call_chat_model")
    
    # Compile the graph
    .compile(
        # name="My Agent Graph",
    )
)
