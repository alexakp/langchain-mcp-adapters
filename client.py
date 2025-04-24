import os
import asyncio


# Create server parameters for stdio connection
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools

endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1-mini"
github_token = os.getenv("github_token")

model = ChatOpenAI(
    openai_api_base=endpoint,
    model_name=model_name,
    openai_api_key=github_token,
    temperature=1.0,
    max_tokens=None, 
)


server_params = StdioServerParameters(
    command="uv",
    args=["run", "math_server.py"],
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "what's (54.54 + 0.1) * 678?"})

            return agent_response


# Run the main function
res = asyncio.run(main())

for m in res["messages"]:
    print(m.pretty_print())
