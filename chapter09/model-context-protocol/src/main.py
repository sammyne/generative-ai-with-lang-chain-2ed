import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import json

from utils import Config

model = Config().new_openai_like()

server_params = StdioServerParameters(
    command="python",
    # Update with the full absolute path to math_server.py
    args=["src/math_server.py"],
)


async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)
            response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            print(response)


if __name__ == "__main__":
    asyncio.run(run_agent())
