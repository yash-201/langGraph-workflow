import os
import sys
import json
import sqlite3
import requests
import hashlib
from pathlib import Path
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
import queue
import threading
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient

# Silence benign JSON-schema conversion warning logs from MCP adapters & Google GenAI
logging.getLogger("langchain_mcp_adapters").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

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

# ----------------- RAG Vectorstore & Retriever Setup -----------------
INDEX_ROOT = Path(__file__).parent / ".indices"
INDEX_ROOT.mkdir(exist_ok=True)

def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "models/gemini-embedding-001",
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    emb = GoogleGenerativeAIEmbeddings(
        model=embed_model_name,
        max_retries=6,
        google_api_key=api_key
    )

    if index_dir.exists() and not force_rebuild:
        return FAISS.load_local(
            str(index_dir),
            emb,
            allow_dangerous_deserialization=True
        )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, emb)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        pdf_path = os.path.join(os.path.dirname(__file__), "..", "langsmith-masterclass-main", "islr.pdf")
        if not os.path.exists(pdf_path):
            pdf_path = os.path.join(os.path.dirname(__file__), "islr.pdf")
        vectorstore = load_or_build_index(pdf_path)
        _retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return _retriever

@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant information from the pdf document.
    Use this tool when the user asks factual / conceptual questions
    that might be answered from the stored documents.
    """
    retriever = get_retriever()
    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        'query': query,
        'context': context,
        'metadata': metadata
    }

local_tools = [search_tool, calculator, get_stock_price, get_owner_portfolio, rag_tool]

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 

_cached_tools = None

async def get_all_tools():
    global _cached_tools
    if _cached_tools is None:
        mcp_tools = await client.get_tools()
        _cached_tools = local_tools + mcp_tools
        print("Loaded tools:", [t.name for t in _cached_tools])
    return _cached_tools

async def build_graph(checkpointer=None):
    all_tools = await get_all_tools()
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
    graph.add_conditional_edges('chat_node', tools_condition, ['tools', END])
    graph.add_edge('tools', 'chat_node')

    workflow = graph.compile(checkpointer=checkpointer)
    
    return workflow

async def main():
    # Open the checkpoint connection asynchronously
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        workflow = await build_graph(checkpointer=checkpointer)
        # CONFIG = {'configurable': {'thread_id': 'thread-1'}}

        # response = await chatbot.ainvoke(
        #     {"messages": [HumanMessage(content="give me the list of expense ")]},
        #     config=CONFIG
        # )
        # final_msg = response['messages'][-1]
        # content = final_msg.content
        # if isinstance(content, list):
        #     text_content = "".join([item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text'])
        # else:
        #     text_content = str(content)
            
        # print("\nFinal Response:\n" + text_content)


class AsyncGraphEngine:
    """
    Persistent Async Graph Engine.
    Maintains a single background worker thread with a persistent event loop,
    and a single AsyncSqliteSaver + Compiled Graph instance.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path=DB_PATH):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    def __init__(self, db_path):
        self.db_path = db_path
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        
        fut = asyncio.run_coroutine_threadsafe(self._init_async(), self.loop)
        fut.result()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _init_async(self):
        self.checkpointer_cm = AsyncSqliteSaver.from_conn_string(self.db_path)
        self.checkpointer = await self.checkpointer_cm.__aenter__()
        self.compiled_graph = await build_graph(self.checkpointer)

    def get_state(self, config):
        async def _get():
            return await self.compiled_graph.aget_state(config)
        fut = asyncio.run_coroutine_threadsafe(_get(), self.loop)
        return fut.result()

    def stream(self, input_data, config=None, stream_mode="messages"):
        q = queue.Queue()
        SENTINEL = object()

        async def _stream():
            try:
                async for chunk, meta in self.compiled_graph.astream(input_data, config=config, stream_mode=stream_mode):
                    q.put((chunk, meta))
            except Exception as e:
                q.put(e)
            finally:
                q.put(SENTINEL)

        asyncio.run_coroutine_threadsafe(_stream(), self.loop)

        while True:
            item = q.get()
            if item is SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item


workflow = AsyncGraphEngine.get_instance()


def retrieve_all_threads():
    all_threads = []
    try:
        conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC")
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
    
