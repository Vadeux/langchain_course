from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from reflecton_agent.nodes import generation_node, reflection_node
from reflecton_agent.utils import MessageGraph

load_dotenv()


def should_continue(state: MessageGraph) -> str:
    if len(state["messages"]) > 6:
        return END
    return REFLECT


REFLECT = "reflect"
GENERATE = "generate"

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)  # place where to start

builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

app = builder.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow_diagram.png")

if __name__ == "__main__":
    print("Hi from langGraph")

    res = app.invoke({"messages": [HumanMessage(content="""
                        Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.
            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.
            Made a video covering their newest blog post
                    """)]})
    print(res["messages"][-1].content)
