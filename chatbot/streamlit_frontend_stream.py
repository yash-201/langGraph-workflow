import streamlit as st
from langgrap_backend import workflow
from langchain_core.messages import HumanMessage

CONFIG = { 'configurable': { 'thread_id': 'thread-1' } }

# st.session_state for maintaining state
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


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
