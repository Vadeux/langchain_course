from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from corrective_RAG.graph.chains.answer_grader import answer_grader
from corrective_RAG.graph.chains.hallucination_grader import hallucination_grader
from corrective_RAG.graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from corrective_RAG.graph.nodes import generate, grade_documents, retrieve, web_search
from corrective_RAG.graph.state import GraphState

load_dotenv()


def decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION---")
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grader_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---Check HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := hallucination_score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        answer_score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if answer_grade := answer_score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESSES QUESTION---")
            return "not userful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE_TRY---")
        return "not supported"


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)

workflow.add_conditional_edges(
    GENERATE,
    grader_generation_grounded_in_documents_and_question,
    {"useful": END, "not useful": WEBSEARCH, "not supported": GENERATE},
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
