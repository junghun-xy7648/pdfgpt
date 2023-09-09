import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

# 시작하기

st.title('chatgpt app')

response = requests.get('http://127.0.0.1:8000/')
# print(response.text) # <Response [200]>은 정상적으로 요청이 들어왔다.

response_data = json.loads(response.text)

st.text(response_data['message'])

question = st.text_input('질문')

if question.strip() != "":
    response = requests.post(
        'http://127.0.0.1:8000/chat', 
        json.dumps({'question' : question})
        )
    
    response_data = json.loads(response.text)
    st.text(response_data['answer'])