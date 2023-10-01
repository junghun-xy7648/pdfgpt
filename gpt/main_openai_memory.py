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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
import pinecone


# Scripts\activate.bat
# uvicorn main:app --reload
# 컨트롤 + C 누르면 서버 멈춤
load_dotenv(verbose=True)

openai.api_key = os.environ["OPENAI_API_KEY"]
pinecone.init(api_key=os.environ["PIENCONE_API_KEY"], environment=os.environ["PIENCONE_ENVIRONMENT"])


print(pinecone.list_indexes())
index = pinecone.Index("steeldb")

# loader = PyPDFLoader('pdf\Principles of Billet Soft-reduction and Consequences for Continuous Casting.pdf') # load file
loader = PyPDFDirectoryLoader('pdf/') # load file
pages = loader.load_and_split() # split document, 디폴트는 페이지 별로 쪼갰음

text_splitter = RecursiveCharacterTextSplitter(
    # # Set a really small chunk size, just to show.
    chunk_size = 1000, # chunk size를 조절해야함
    chunk_overlap  = 100,
    length_function = len,
    add_start_index = True,
)
texts = text_splitter.split_documents(pages)
print("=" * 50)
# print([document.page_content for document in texts])
print(texts)
print("=" * 50)
# print(texts[0])
# print(texts[1])

print(len(texts))

# page_contents = [document.page_content for document in texts]

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

for document in texts:
    embed = get_embedding(document.page_content, model='text-embedding-ada-002')
    
    
    print(len(embed))
    index.upsert(
        vectors=[("A"+f'{document.metadata["start_index"]}', embed, document.metadata)],
    )



print(index.describe_index_stats())

db = Chroma.from_documents(texts, OpenAIEmbeddings(), persist_directory="./vectorstore") # embedding -> vertor store

class QuestionRequest(BaseModel):
    question: str # question의 값은 str이어야 함

app = FastAPI()

# llm = OpenAI(temperature=0, max_tokens=-1) # 1에 가까울수록 할루시네이션 허용 0이면 PDF 내용만 설명
# prompt_template = PromptTemplate(
#     input_variables=["input_documents", "chat_history", "human_input"],
#     template="""
# AI assistant is a brand new, powerful, human-like artificial intelligence.
# The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness.
# AI is a well-behaved and well-mannered individual.
# AI is always friendly, kind, and inspiring, and he is eager to provide vivid and thoughtful responses to the user.
# AI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation.
# AI assistant is a big fan of Pinecone and Vercel.
# START CONTEXT BLOCK
# [context]
# {input_documents}
# END OF CONTEXT BLOCK
# AI assistant will take into account any CONTEXT BLOCK that is provided in a conversation.
# If the context does not provide the answer to question, the AI assistant will say, "I'm sorry, but I don't know the answer to that question".
# AI assistant will not apologize for previous responses, but instead will indicated new information was gained.
# AI assistant will not invent anything that is not drawn directly from the context.
# Please answer in English.

# Previous conversation:
# {chat_history}
# Human: {human_input}
# """
# )

# Notice that we need to align the `memory_key`
# memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="human_input", k=5)
# memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

# chain = load_qa_chain(llm=llm, prompt=prompt_template, verbose=True, memory=memory)

chat_history = []

enter = "\n"

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(request_body: QuestionRequest): # fastapi 규칙
    # print(request_body.question)
    query = request_body.question
    print(query)
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
    # response = chain({
    #                     'input_documents': "\n".join(page_contents),
    #                     "human_input" : query # or whatever you want
    #                     }, return_only_outputs=True)
    
    messages=[
            {"role": "user", "content": f""" AI assistant is a brand new, powerful, human-like artificial intelligence.
    The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness.
    AI is a well-behaved and well-mannered individual.
    AI is always friendly, kind, and inspiring, and he is eager to provide vivid and thoughtful responses to the user.
    AI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation.
    AI assistant is a big fan of Pinecone and Vercel.
    START CONTEXT BLOCK
    [context]
    {f"{enter}".join(page_contents)}
    END OF CONTEXT BLOCK
    AI assistant will take into account any CONTEXT BLOCK that is provided in a conversation.
    If the context does not provide the answer to question, the AI assistant will say, "I'm sorry, but I don't know the answer to that question".
    AI assistant will not apologize for previous responses, but instead will indicated new information was gained.
    AI assistant will not invent anything that is not drawn directly from the context.
    Please answer in English.

    Previous conversation:
    {f"{enter}".join([f"{chat['role']}: {chat['content']}" for chat in chat_history[-6:]])}
    user: {query}"""}
        ]
    
    print(messages)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    
   

    # Run the chain only specifying the input variable.
    # 질문을 openai에게 준다.request_body.question가 query임


    # print(answer)
    # answer = response['choices'][0]['message']['content'] # openai인 경우의 chatgpt의 답변이다.
    
    return {"answer": response['choices'][0]['message']['content']}