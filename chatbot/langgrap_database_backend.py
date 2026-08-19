from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
import operator
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv(override=True)

# Initialize Gemini Model
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
# Generator: Production model for complex creative/technical output
model = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash',  # gemini-3.6-flash # gemini-3.5-flash
    max_retries=6,
    google_api_key=api_key
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 



def chat_node(state: ChatState):
    messages = state['messages']

    response = model.invoke(messages) # entire output return 
    
    return { 'messages': [response]}

DB_PATH = os.path.join(os.path.dirname(__file__), "chatbot.db")
conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

#add node
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

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

    
# for message_chunk, metadata in workflow.stream(
#     { "messages": [HumanMessage(content="Hello")] },
#     config={'configurable': {'thread_id': 'thread-1'}},
#     stream_mode="messages"
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end='', flush=True)

# CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# response = workflow.invoke({ "messages": [HumanMessage(content="what is my name")] }, config=CONFIG)

# print("response ", response)
# print(workflow.get_state(config=CONFIG).values['messages'])
