from dotenv import load_dotenv
from graph.graph import app

load_dotenv()

if __name__ == "__main__":
    print("Hi from corrective RAG agent")
    print(app.invoke(input={"question": "What is agent memory?"}))
    # print(app.invoke(input={"question": "How to make pizza?"}))
