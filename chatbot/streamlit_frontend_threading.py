import streamlit as st
from langgrap_backend import workflow
from langchain_core.messages import HumanMessage
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

# *********************** utility functions end **********************

# ********************** session setup start **********************
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = create_new_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = []

add_thread(st.session_state.thread_id)
# ********************** session setup end **********************

# ********************** sidebar setup start **********************
st.sidebar.title("LangGraph Bot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Conversation History")
for thread_id in st.session_state.chat_threads[::-1]:
    if st.sidebar.button(thread_id, key=thread_id):
        st.session_state.thread_id = thread_id
        messages = load_conversion(thread_id)

        temp_messages = []
        for m in messages:
            role = 'user' if isinstance(m, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': extract_text(m.content)})
        st.session_state.message_history = temp_messages
        st.rerun()

# ********************** sidebar setup end **********************

CONFIG = {'configurable': {'thread_id': st.session_state.thread_id}}

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
            content = message_chunk.content
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        yield block["text"]
                    elif isinstance(block, str):
                        yield block

    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            ai_message = st.write_stream(stream_generator())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

