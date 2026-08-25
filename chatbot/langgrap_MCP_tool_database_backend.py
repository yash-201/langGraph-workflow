import os
import sys
import sqlite3
import requests
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(override=True)

DB_PATH = os.path.join(os.path.dirname(__file__), "chatbot.db")

# Initialize Gemini Model
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    model='gemini-3.6-flash',
    max_retries=6,
    google_api_key=api_key
)

client = MultiServerMCPClient(
    {
        "fastmcp": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                "server.py"
            ],
            "cwd": r"D:\expense-tracker-mcp-server"
        }
    }
)

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic operations (add, sub, mul, div) on two numbers."""
    try:
        if operation == "add":
            res = first_num + second_num
        elif operation == "sub":
            res = first_num - second_num
        elif operation == "mul":
            res = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero"}
            res = first_num / second_num
        else:
            return {"error": "Invalid operation"}
    
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": res
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=O2F1U7F7Y438LGNM"
    r = requests.get(url)
    return r.json()

@tool
def get_owner_portfolio():
    """
    Return the owner information which is developed this product
    """
    return {"name": "Yash", "age": 22, "company": "Self"}

local_tools = [search_tool, calculator, get_stock_price, get_owner_portfolio]

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 

async def build_graph(checkpointer=None):
    mcp_tools = await client.get_tools()
    all_tools = local_tools + mcp_tools

    print("mcp tools: ", all_tools)
    llm_with_tools = model.bind_tools(all_tools)
    
    async def chat_node(state: ChatState):
        messages = state['messages']
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(all_tools)

    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node('chat_node', chat_node)
    graph.add_node('tools', tool_node)

    # Add edges
    graph.add_edge(START, 'chat_node')
    graph.add_conditional_edges('chat_node', tools_condition, ['tools', END]) # here END is keyword to stop the execution of the graph.
    graph.add_edge('tools', 'chat_node')

    workflow = graph.compile(checkpointer=checkpointer)
    
    return workflow

async def main():
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        chatbot = await build_graph(checkpointer=checkpointer)
        CONFIG = {'configurable': {'thread_id': 'thread-1'}}

        response = await chatbot.ainvoke(
            {"messages": [HumanMessage(content="give me the list of expense ")]},
            config=CONFIG
        )
        final_msg = response['messages'][-1]
        content = final_msg.content
        if isinstance(content, list):
            text_content = "".join([item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text'])
        else:
            text_content = str(content)
            
        print("\nFinal Response:\n" + text_content)


def retrieve_all_threads():
    all_threads = []
    try:
        conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        all_threads = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
    except Exception as e:
        print(f"Error retrieving threads: {e}")
    return all_threads


if __name__ == "__main__":
    asyncio.run(main())
#     CONFIG = {'configurable': {'thread_id': 'thread-1'}}
#     response = workflow.invoke(
#         {"messages": [HumanMessage(content="what is the stock price of apple? how much doller i need to purchase 3 share ")]},
#         config=CONFIG
#     )
#     # print("response ", response)
#     # print(workflow.get_state(config=CONFIG).values['messages'])
#     print(response['messages'][-1].content)
    
