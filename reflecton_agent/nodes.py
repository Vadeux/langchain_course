from chains import generation_chain, reflect_chain
from langchain_core.messages import HumanMessage

from reflecton_agent.utils import MessageGraph


def generation_node(state: MessageGraph) -> MessageGraph:
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph) -> MessageGraph:
    result = reflect_chain.invoke({"messages": state["messages"]})
    return {
        "messages": [HumanMessage(content=result.content)]
    }  # We wrap the result in a HumanMessage because we put it to the llm
