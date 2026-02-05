import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Initialize embeddings (same as in ingestion.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME_DOC_HELPER"], embedding=embeddings
)

# Initialize chat model
model = init_chat_model("gpt-4o", model_provider="openai")


@tool(response_format="content_and_artifact")  # return 2 values (content_and_artifact)
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Get top-4 docs relevant
    retrieved_docs = vector_store.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents (because content_and_artifact)
    # serialized content will go to the LLM, and raw docs will be used in somewhere in app logic
    return serialized, retrieved_docs


def run_llm(query: str) -> dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    Args:
        query: The user's question.
    Returns:
        Dictionary containing:
            - answer: The generated answer.
            - context: List of retrieved documents.
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list
    messages = [
        {"role": "user", "content": query},
    ]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    result = run_llm("What are deep agents?")
    print(result)
