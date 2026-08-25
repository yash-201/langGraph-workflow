import streamlit as st
from langgrap_MCP_tool_database_backend import workflow, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# *********************** utility functions start **********************
def create_new_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    st.session_state.message_history = []
    st.session_state.thread_id = create_new_thread_id()
    add_thread(st.session_state.thread_id)
    st.rerun()

def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

def extract_text(content):
    if isinstance(content, list):
        return "".join(
            block['text'] if isinstance(block, dict) and 'text' in block else str(block)
            for block in content
        )
    return str(content)

def load_conversion(thread_id):
    """load conversation from state safely using .get()"""
    state = workflow.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def get_thread_title(thread_id, messages=None):
    if messages is None:
        messages = load_conversion(thread_id)
    for m in messages:
        if isinstance(m, HumanMessage):
            text = extract_text(m.content).strip()
            if text:
                return text[:30] + "..." if len(text) > 30 else text
    return f"New Chat ({thread_id[:8]})"

# *********************** utility functions end **********************

# ********************** session setup start **********************
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = create_new_thread_id()

if 'thread_titles' not in st.session_state:
    st.session_state.thread_titles = {}

db_threads = retrieve_all_threads()
if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = db_threads
else:
    for t in db_threads:
        if t not in st.session_state.chat_threads:
            st.session_state.chat_threads.append(t)

add_thread(st.session_state.thread_id)
# ********************** session setup end **********************

# ********************** sidebar setup start **********************
st.sidebar.title("LangGraph Bot")

if st.sidebar.button("New Chat"):
    reset_chat()

# Update active thread title if user sent messages
if st.session_state.message_history and st.session_state.thread_id not in st.session_state.thread_titles:
    for msg in st.session_state.message_history:
        if msg.get('role') == 'user' and msg.get('content'):
            text = str(msg['content']).strip()
            st.session_state.thread_titles[st.session_state.thread_id] = text[:30] + "..." if len(text) > 30 else text
            break

st.sidebar.header("Conversation History")
for thread_id in st.session_state.chat_threads[::-1]:
    title = st.session_state.thread_titles.get(thread_id, f"Chat ({thread_id[:8]})")

    if st.sidebar.button(title, key=thread_id):
        st.session_state.thread_id = thread_id
        messages = load_conversion(thread_id)
        temp_messages = []
        pending_tools = []
        for m in messages:
            if isinstance(m, HumanMessage):
                temp_messages.append({'role': 'user', 'content': extract_text(m.content)})
                pending_tools = []
            elif isinstance(m, ToolMessage) and getattr(m, 'name', None):
                if m.name not in pending_tools:
                    pending_tools.append(m.name)
            elif isinstance(m, AIMessage):
                if getattr(m, 'tool_calls', None):
                    for tc in m.tool_calls:
                        t_name = tc.get('name')
                        if t_name and t_name not in pending_tools:
                            pending_tools.append(t_name)
                elif extract_text(m.content).strip():
                    msg_obj = {'role': 'assistant', 'content': extract_text(m.content)}
                    if pending_tools:
                        msg_obj['tools_used'] = list(pending_tools)
                        pending_tools = []
                    temp_messages.append(msg_obj)
        st.session_state.message_history = temp_messages
        if temp_messages:
            for msg in temp_messages:
                if msg.get('role') == 'user':
                    text = msg['content'].strip()
                    st.session_state.thread_titles[thread_id] = text[:30] + "..." if len(text) > 30 else text
                    break
        st.rerun()

# ********************** sidebar setup end **********************

CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {
        "thread_id": st.session_state["thread_id"]
    },
    "run_name": "chat_turn",
}

# ********************** display existing history start **********************
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        if message.get('tools_used'):
            st.caption("🛠️ **Tools used:** " + ", ".join([f"`{t}`" for t in message['tools_used']]))
        st.markdown(message['content'])
# ********************** display existing history end **********************


user_input = st.chat_input("Ask a question")

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    

    called_tools = []

    def stream_generator():
        for message_chunk, metadata in workflow.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages"
        ):
            # Track tools executed
            if isinstance(message_chunk, ToolMessage) and getattr(message_chunk, 'name', None):
                if message_chunk.name not in called_tools:
                    called_tools.append(message_chunk.name)
            elif isinstance(message_chunk, AIMessage) and getattr(message_chunk, 'tool_calls', None):
                for tc in message_chunk.tool_calls:
                    t_name = tc.get('name')
                    if t_name and t_name not in called_tools:
                        called_tools.append(t_name)

            # Only stream final text responses from AIMessage (ignore raw ToolMessages & tool call requests)
            if isinstance(message_chunk, AIMessage) and not getattr(message_chunk, 'tool_calls', None):
                content = message_chunk.content
                if isinstance(content, str) and content:
                    yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            yield block["text"]
                        elif isinstance(block, str) and block:
                            yield block

    with st.chat_message('assistant'):
        with st.spinner("Thinking & executing tools..."):
            ai_message = st.write_stream(stream_generator())
        if called_tools:
            st.caption("🛠️ **Tools used:** " + ", ".join([f"`{t}`" for t in called_tools]))

    if ai_message:
        msg_obj = {'role': 'assistant', 'content': ai_message}
        if called_tools:
            msg_obj['tools_used'] = called_tools
        st.session_state['message_history'].append(msg_obj)
    add_thread(st.session_state.thread_id)
    st.rerun()

