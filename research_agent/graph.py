from langgraph.constants import END
from langgraph.graph import StateGraph

from research_agent.const import (
    NODE_COMPILE_REPORT,
    NODE_CONDUCT_RESEARCH,
    NODE_GENERATE_SUBTOPICS,
    NODE_GET_APPROVAL,
    ApprovalState,
)
from research_agent.nodes import (
    ask_approval,
    compile_report,
    conduct_research,
    generate_subtopics,
)
from research_agent.schemas import GraphState


def route_approval(state: GraphState) -> str:
    if state["approval"] == ApprovalState.APPROVED.value:
        print("--- SUBTOPICS APPROVED ---")
        return NODE_CONDUCT_RESEARCH
    print("--- SUBTOPICS REJECTED. RETURNING TO SUBTOPIC GENERATION ---")
    return NODE_GENERATE_SUBTOPICS


builder = StateGraph(GraphState)

builder.add_node(NODE_GENERATE_SUBTOPICS, generate_subtopics)
builder.add_node(NODE_GET_APPROVAL, ask_approval)
builder.add_node(NODE_CONDUCT_RESEARCH, conduct_research)
builder.add_node(NODE_COMPILE_REPORT, compile_report)

builder.set_entry_point(NODE_GENERATE_SUBTOPICS)

builder.add_edge(NODE_GENERATE_SUBTOPICS, NODE_GET_APPROVAL)
builder.add_conditional_edges(
    NODE_GET_APPROVAL,
    route_approval,
    {
        NODE_GENERATE_SUBTOPICS: NODE_GENERATE_SUBTOPICS,
        NODE_CONDUCT_RESEARCH: NODE_CONDUCT_RESEARCH,
    },
)

builder.add_edge(NODE_CONDUCT_RESEARCH, NODE_COMPILE_REPORT)
builder.add_edge(NODE_COMPILE_REPORT, END)

research_agent = builder.compile()
research_agent.get_graph().draw_mermaid_png(output_file_path="graph.png")
