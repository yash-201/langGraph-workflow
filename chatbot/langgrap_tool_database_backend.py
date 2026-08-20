import os
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
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(override=True)

# Initialize Gemini Model
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash',
    max_retries=6,
    google_api_key=api_key
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

tools = [search_tool, calculator, get_stock_price, get_owner_portfolio]

llm_with_tools = model.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

DB_PATH = os.path.join(os.path.dirname(__file__), "chatbot.db")
conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

# Add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# Add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition, ['tools', END])
graph.add_edge('tools', 'chat_node')

workflow = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = []
    try:
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config.get('configurable', {}).get('thread_id')
            if thread_id and thread_id not in all_threads:
                all_threads.append(thread_id)
    except Exception as e:
        print(f"Error retrieving threads: {e}")
    return all_threads


# if __name__ == "__main__":
#     CONFIG = {'configurable': {'thread_id': 'thread-1'}}
#     response = workflow.invoke(
#         {"messages": [HumanMessage(content="what is the stock price of apple? how much doller i need to purchase 3 share ")]},
#         config=CONFIG
#     )
#     # print("response ", response)
#     # print(workflow.get_state(config=CONFIG).values['messages'])
#     print(response['messages'][-1].content)
    
