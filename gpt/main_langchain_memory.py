from fastapi import FastAPI
from pydantic import BaseModel
import openai
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# uvicorn main:app --reload
# 컨트롤 + C 누르면 서버 멈춤
load_dotenv(verbose=True)

# loader = PyPDFLoader('pdf\Principles of Billet Soft-reduction and Consequences for Continuous Casting.pdf') # load file
loader = PyPDFDirectoryLoader('pdf/') # load file
pages = loader.load_and_split() # split document, 디폴트는 페이지 별로 쪼갰음

text_splitter = CharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 3000, # chunk size를 조절해야함
    chunk_overlap  = 0,
    # length_function = len,
    # add_start_index = True,
)
texts = text_splitter.split_documents(pages)
# print(texts[0])
# print(texts[1])

db = Chroma.from_documents(texts, OpenAIEmbeddings(), persist_directory="./vectorstore") # embedding -> vertor store
print('db:', db)
class QuestionRequest(BaseModel):
    question: str # question의 값은 str이어야 함

app = FastAPI()

llm = OpenAI(temperature=0, max_tokens=-1) # 1에 가까울수록 할루시네이션 허용 0이면 PDF 내용만 설명
prompt_template = PromptTemplate(
    input_variables=["context", "chat_history", "human_input"],
    template="""
=====================================================================
You are a steel engineer who answers questions about PDFs when they come in.
[query] Find the question within [context] and explain it kindly and accurately.
Answer [query] in context.
Please explain so that quantitative figures are included rather than qualitative expressions.
If there's anything you don't know, "I don't know. Ask me another question."
Please answer in Korean
=====================================================================
[context]
{context}
=====================================================================
[query]
Previous conversation:
{chat_history}
Human: {human_input}
=====================================================================
"""
)

# Notice that we need to align the `memory_key`
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")


chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory, verbose=True)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(request_body: QuestionRequest): # fastapi 규칙
    # print(request_body.question)
    query = request_body.question
    embedding_vector = OpenAIEmbeddings().embed_query(query)
    # searched = db.similarity_search(query, k=5) # k:3 질문과 유사한거 3개 찾아라, pdf page 3장을 의미함
    searched = db.similarity_search_by_vector(embedding_vector, k=5)
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo", # gpt-4-0613 # gpt-3.5-turbo
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": prompt_template.format(context=searched, query=query)
    #         }, # 질문을 openai에게 준다. request_body.question가 query임
    #     ]
    # )
    page_contents = [document.page_content for document in searched]
    print("page_contents: ", page_contents)
    response = chain.run({
                        'context': page_contents[0],
                        "chat_history" : memory,
                        "human_input" : query # or whatever you want
                        })
    # print(response)

    # Run the chain only specifying the input variable.
    # 질문을 openai에게 준다.request_body.question가 query임


    # print(answer)
    # answer = response['choices'][0]['message']['content'] # openai인 경우의 chatgpt의 답변이다.
    
    return {"answer": response}