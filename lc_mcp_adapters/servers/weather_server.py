from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"Weather in {city}: sunny!"


if __name__ == "__main__":
    mcp.run(transport="sse")
