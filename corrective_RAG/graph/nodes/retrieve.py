from typing import Any

from corrective_RAG.graph.state import GraphState
from corrective_RAG.ingestion import retriever


def retrieve(state: GraphState) -> dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
