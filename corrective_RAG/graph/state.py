import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: Annotated[list[Document], operator.add]  # list[str]
