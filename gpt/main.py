from fastapi import FastAPI
from pydantic import BaseModel
import openai

openai.api_key = "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"

class QuestionRequest(BaseModel):
    question: str # question의 값은 str이어야 함

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(request_body: QuestionRequest): # fastapi 규칙
    # print(request_body.question)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": request_body.question}, # 질문을 openai에게 준다.
        ]
    )
    # print(answer)
    answer = response['choices'][0]['message']['content'] # chatgpt의 답변이다.
    return {"answer": answer}