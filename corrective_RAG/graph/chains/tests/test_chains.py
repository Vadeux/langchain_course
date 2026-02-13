from pprint import pprint

from dotenv import load_dotenv

from corrective_RAG.graph.chains.generation import generation_chain
from corrective_RAG.graph.chains.retrieval_grader import (
    GradeDocuments,
    retrieval_grader,
)
from corrective_RAG.ingestion import retriever

load_dotenv()


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    result = GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert result.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    result = GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizza", "document": doc_txt}
    )
    assert result.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    pprint(generation)
