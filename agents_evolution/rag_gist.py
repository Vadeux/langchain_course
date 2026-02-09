import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("../utils/mediumblog1.txt")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(document)
    print(f"Created {len(text)} chunks.")

    embeddings = OpenAIEmbeddings()

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        text, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("Done!")
