import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters, stdio_client

load_dotenv()
llm = ChatOpenAI()

stdio_server_params = StdioServerParameters(
    command="python",
    args=[
        "/Users/vadim/study/PythonProject/langchain_course/lc_mcp_adapters/servers/math_server.py"
    ],
)


async def main():
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("--- Session initialized ---")
            tools = await load_mcp_tools(session)
            print(tools)
            agent = create_agent(llm, tools)

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is 2 + 2 * 2?")]}
            )
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
