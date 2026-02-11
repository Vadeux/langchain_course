from typing import Literal

from chains import first_responder, revisor
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from tool_executer import execute_tools

load_dotenv()

MAX_ITERATIONS = 2


def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count."""
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges(
    "revise", event_loop, {END: END, "execute_tools": "execute_tools"}
)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="flow_reflexion.png")

if __name__ == "__main__":
    print("Hi reflexion agent.")
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
                }
            ]
        }
    )
    last_message = result["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(last_message.tool_calls[0]["args"]["answer"])
    print(result)
