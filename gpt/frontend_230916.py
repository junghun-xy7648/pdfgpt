import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

# streamlit run frontend.py
# 시작하기
# response = requests.get('http://127.0.0.1:8000/')
# print(response.text) # <Response [200]>은 정상적으로 요청이 들어왔다.
# response_data = json.loads(response.text)
# st.text(response_data['message'])

# 채팅시스템 추가
# 초기
# st.session_state = {'chat_history' : []}
# 대화 입력 후
# st.session_state = {'chat_history' : [{'role': 'user', 'message': question}]} 
# 그러므로 st.session_state.chat_history에 대화를 저장해야함

# 파일 업로더

st.title('chatgpt app')


if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [] # 최초 초기화

question = st.chat_input('질문을 적으세요.')

if question != None and question.strip() != "":
    st.session_state.chat_history.append({'role': 'user', 'message': question})
    response = requests.post(
        'http://127.0.0.1:8000/chat', 
        json.dumps({'question' : question})
        )
    
    response_data = json.loads(response.text)
    st.session_state.chat_history.append({'role': 'assistant', 'message': response_data['answer']})
    
with st.container():
    for chat_message in st.session_state.chat_history:
        if chat_message['role'] == 'user':
            user = st.chat_message('user')
            user.write(chat_message['message'])
        elif chat_message['role'] == 'assistant':
            assistant = st.chat_message('assistant')
            assistant.write(chat_message['message'])