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

# uvicorn main:app --reload
load_dotenv(verbose=True)

# loader = PyPDFLoader('pdf\Principles of Billet Soft-reduction and Consequences for Continuous Casting.pdf') # load file
loader = PyPDFDirectoryLoader('pdf/') # load file
pages = loader.load_and_split() # split document
db = Chroma.from_documents(pages, OpenAIEmbeddings(), persist_directory="./vectorstore") # embedding -> vertor store

class QuestionRequest(BaseModel):
    question: str # question의 값은 str이어야 함

app = FastAPI()

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="""You are a steel engineer who answers questions about PDFs when they come in.
                [query] Find the question within [context] and explain it kindly and accurately.
                Please explain so that quantitative figures are included rather than qualitative expressions.
                If there's anything you don't know, "I don't know. Ask me another question."
                Please answer in Korean
                =====================================================================
                [context]
                {context}
                
                =====================================================================
                [query]
                {query}
                """
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(request_body: QuestionRequest): # fastapi 규칙
    # print(request_body.question)
    query = request_body.question
    
    searched = db.similarity_search(query, k=3) # k:3 질문과 유사한거 3개 찾아라
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", # gpt-4-0613 # gpt-3.5-turbo
        messages=[
            {
                "role": "user",
                "content": prompt_template.format(context=searched, query=query)
            }, # 질문을 openai에게 준다. request_body.question가 query임
        ]
    )
    # print(answer)
    answer = response['choices'][0]['message']['content'] # chatgpt의 답변이다.
    return {"answer": answer}