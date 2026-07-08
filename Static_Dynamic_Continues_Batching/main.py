from fastapi import FastAPI
import uvicorn

from model import Chat

app = FastAPI()

chat = Chat()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/chat/{message}")
async def chat_endpoint(message: str):
    await chat.add_message("user", message)
    response = await chat.get_response()
    await chat.add_message("assistant", response)
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, port=8000)