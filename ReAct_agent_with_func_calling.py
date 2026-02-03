from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Return the length of a text by characters."""
    print(f"get_text_length called with: {text}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: list[BaseTool], tool_name: str) -> BaseTool | ValueError:
    """Get python func by name from tools, which we can .invoke()."""
    for t in tools:
        if t.name == tool_name:
            return t
    return ValueError(f"Tool with name {tool_name} not found")


def main():
    print("Hello from ReAct LangChain!")
    tools = [get_text_length]

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4",
        callbacks=[AgentCallbackHandler()],
    )

    llm_with_tools = llm.bind_tools(tools)

    messages: list = [
        HumanMessage(content="What is the length of the word: DOG")
    ]  # list, where me collect all conversation history

    while True:
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                # tool_call is a dict with keys: name, args, id[, type]
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                call_tool_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)  # why not **?
                print(f"Observation: {observation}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=call_tool_id)
                )
            continue

        print(ai_message.content)
        break


if __name__ == "__main__":
    main()
