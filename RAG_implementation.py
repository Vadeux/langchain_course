import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model="gpt-4o")

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # top-3 relevant docs

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context
    
    {context}
    
    Question: {question}
    
    Provide a detailed answer:"""
)


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chai_without_lcel(query: str) -> str:
    """
    Simple retrieval chain without langchain expression language.
    Manually retrieves docs, formats them, and generates a response.

     Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format retrieved documents into a single context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke llm with the formatted messages
    response = llm.invoke(messages)
    return response.content


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LangChain Expression Language (LCEL).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retriever_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )  # lc will auto convert py func into RunnableLambda(format_docs)
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retriever_chain


if __name__ == "__main__":
    print("Retrieving...")

    query = "What is Pinecone in machine learning?"

    # result_without_lcel = retrieval_chai_without_lcel(query)
    # print(result_without_lcel)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result = chain_with_lcel.invoke({"question": query})
    print(result)
