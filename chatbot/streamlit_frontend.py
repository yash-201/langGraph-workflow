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
    

    response = workflow.invoke({ 'messages': [HumanMessage(content=user_input)] }, config=CONFIG)
    raw_content = response['messages'][-1].content
    
    if isinstance(raw_content, list):
        print("is list")
        ai_message = "".join(
            block['text'] if isinstance(block, dict) and 'text' in block else str(block)
            for block in raw_content
        )
    else:
        print("is not list")
        ai_message = str(raw_content)

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.markdown(ai_message)
