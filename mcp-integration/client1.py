import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage
import json

load_dotenv()


api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

SERVERS = { 
    # "math": {
    #     "transport": "stdio",
    #     "command": "/Library/Frameworks/Python.framework/Versions/3.11/bin/uv",
    #     "args": [
    #         "run",
    #         "fastmcp",
    #         "run",
    #         "/Users/nitish/Desktop/mcp-math-server/main.py"
    #    ]
    # },
    "expense": {
        "transport": "streamable_http",  # if this fails, try "sse"
        "url": "https://expense-tracker-ykp.fastmcp.app/mcp"
    },
    # "manim-server": {
    #     "transport": "stdio",
    #     "command": "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
    #     "args": [
    #     "/Users/nitish/desktop/manim-mcp-server/src/manim_server.py"
    #   ],
    #     "env": {
    #     "MANIM_EXECUTABLE": "/Library/Frameworks/Python.framework/Versions/3.11/bin/manim"
    #   }
    # }
}

async def main():
    
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()


    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool

    print("Available tools:", named_tools.keys())

    llm = ChatGoogleGenerativeAI(
        model='gemini-3.5-flash',
        max_retries=6,
        google_api_key=api_key
    )
    llm_with_tools = llm.bind_tools(tools)

    prompt = "add expense amount 1000 date today category food note lunch"
    response = await llm_with_tools.ainvoke(prompt)

    if not getattr(response, "tool_calls", None):
        print("\nLLM Reply:", response.content)
        return

    tool_messages = []
    for tc in response.tool_calls:
        selected_tool = tc["name"]
        selected_tool_args = tc.get("args") or {}
        selected_tool_id = tc["id"]

        result = await named_tools[selected_tool].ainvoke(selected_tool_args)
        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(result)))
        

    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])
    print(f"Final response: {final_response.content}")


if __name__ == '__main__':
    asyncio.run(main())