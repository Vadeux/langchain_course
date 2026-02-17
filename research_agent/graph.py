from langgraph.constants import END
from langgraph.graph import StateGraph

from research_agent.nodes import (
    ask_approval,
    compile_report,
    conduct_research,
    generate_subtopics,
)
from research_agent.schemas import GraphState


def route_approval(state: GraphState) -> str:
    if state["approval"] == "approved":
        return "conduct_research"
    return "generate_subtopics"


builder = StateGraph(GraphState)

builder.add_node("generate_subtopics", generate_subtopics)
builder.add_node("get_approval", ask_approval)
builder.add_node("conduct_research", conduct_research)
builder.add_node("compile_report", compile_report)

builder.set_entry_point("generate_subtopics")

builder.add_edge("generate_subtopics", "get_approval")
builder.add_conditional_edges(
    "get_approval",
    route_approval,
    {
        "conduct_research": "conduct_research",
        "generate_subtopics": "generate_subtopics",
    },
)

builder.add_edge("conduct_research", "compile_report")
builder.add_edge("compile_report", END)

research_agent = builder.compile()
research_agent.get_graph().draw_mermaid_png(output_file_path="graph.png")
