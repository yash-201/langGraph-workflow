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
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)

# Initialize Gemini Model
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
# Generator: Production model for complex creative/technical output
model = ChatGoogleGenerativeAI(
    model='gemini-3.6-flash',  # gemini-3.6-flash
    max_retries=6,
    google_api_key=api_key
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 



def chat_node(state: ChatState):
    messages = state['messages']

    response = model.invoke(messages)
    
    return { 'messages': [response]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

#add node
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

workflow = graph.compile(checkpointer=checkpointer)