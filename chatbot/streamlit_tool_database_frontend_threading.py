import streamlit as st
from langgrap_tool_database_backend import workflow, retrieve_all_threads
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

def get_thread_title(thread_id):
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

st.sidebar.header("Conversation History")
for thread_id in st.session_state.chat_threads[::-1]:
    messages = load_conversion(thread_id)
    if not messages and thread_id != st.session_state.thread_id:
        continue

    title = get_thread_title(thread_id)
    if st.sidebar.button(title, key=thread_id):
        st.session_state.thread_id = thread_id
        temp_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                temp_messages.append({'role': 'user', 'content': extract_text(m.content)})
            elif isinstance(m, AIMessage) and extract_text(m.content).strip():
                # Filter out intermediate tool-calling AIMessages
                if not getattr(m, 'tool_calls', None):
                    temp_messages.append({'role': 'assistant', 'content': extract_text(m.content)})
        st.session_state.message_history = temp_messages
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
        st.markdown(message['content'])
# ********************** display existing history end **********************


user_input = st.chat_input("Ask a question")

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    

    def stream_generator():
        for message_chunk, metadata in workflow.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages"
        ):
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

    if ai_message:
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    add_thread(st.session_state.thread_id)
    st.rerun()

