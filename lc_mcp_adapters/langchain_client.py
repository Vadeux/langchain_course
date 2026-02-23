import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI()


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [
                    "/Users/vadim/study/PythonProject/langchain_course/lc_mcp_adapters/servers/math_server.py"
                ],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
        }
    )
    tools = await client.get_tools()
    agent = create_agent(llm, tools)
    # result = await agent.ainvoke({"messages": "What is 2 + 2 * 2?"})
    result = await agent.ainvoke({"messages": "What is the weather in San Francisco?"})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
