from typing import Any

from corrective_RAG.graph.chains.generation import generation_chain
from corrective_RAG.graph.state import GraphState


def generate(state: GraphState) -> dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    result = generation_chain.invoke({"question": question, "context": documents})
    return {"question": question, "documents": documents, "generation": result}
