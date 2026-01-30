from typing import Any, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


# from langchain.tools import tool
# from tavily import TavilyClient
# tavily = TavilyClient()
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over the internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)


llm = ChatOpenAI(temperature=0, model="gpt-5-mini")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from lc course")
    result = cast(Any, agent).invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for 3 job postings for an ai engineer using langchain in the Minsk area on linkedin and list their details"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
