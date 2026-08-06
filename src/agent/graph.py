"""Agent graph definition."""

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import call_model, call_tool, call_tool_or_stop, blog
from agent.state import Context, State
from agent.tools.blog import search_blog_posts

graph = (
    StateGraph(State, context_schema=Context)
    
    # Add nodes to the graph
    .add_node("call_model", call_model)
    .add_node("call_tool", call_tool)
    .add_node("call_blog_tool", ToolNode([search_blog_posts]))
    .add_node("generate_answer", blog.generate_answer)
    .add_node("rewrite_question", blog.rewrite_question)
    
    # Add edges to define the flow of the graph
    .add_edge("__start__", "call_model")
    .add_conditional_edges("call_model", call_tool_or_stop, ["call_tool", "call_blog_tool", "__end__"])
    .add_edge("call_tool", "call_model")
    .add_conditional_edges("call_blog_tool", blog.grade_documents, ["rewrite_question", "generate_answer"])
    .add_edge("rewrite_question", "call_model")
    .add_edge("generate_answer", "__end__")
    
    # Compile the graph
    .compile(
        # name="My Agent Graph",
    )
)
