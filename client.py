from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio

load_dotenv()


async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
            "args":["mathserver.py"],
            "transport":"stdio",
            },
            "weather":{
                "url":"http://127.0.0.1:8000/mcp",
                "transport":"streamable_http",
            }

        }
    )

    tools=await client.get_tools()
    model=ChatOpenAI()
    agent=create_react_agent(
        model,tools
    )

    math_response=await agent.ainvoke(
        {"messages": [{"role":"user","content":"what's (3+5) X 12"}]}
    )
    print("Math response:",math_response['messages'][-1].content)

    weather_response=await agent.ainvoke(
        {"messages": [{"role":"user","content":"what's the weather of kestopur"}]}
    )
    print("Weather response:",weather_response['messages'][-1].content)
    

asyncio.run(main())
