from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from corrective_RAG.graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    if "documents" in state:  # if the route to web search in first time then give error
        documents = state.get("documents", [])
    else:
        documents = None
    tavily_results = web_search_tool.invoke(question)["results"]
    joined_tavily_result = "\n".join(
        [tavily_doc["content"] for tavily_doc in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    if documents is None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
