from typing import Any

from corrective_RAG.graph.chains.retrieval_grader import retrieval_grader
from corrective_RAG.graph.state import GraphState


def grade_documents(state: GraphState) -> dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant, we will ser a flag to run web search.

    Args:
        state (dict): The current graph state.

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state.
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        if score.binary_score.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "web_search": web_search, "question": question}
