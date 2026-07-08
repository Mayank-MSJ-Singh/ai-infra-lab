from fastapi import FastAPI
import uvicorn

from app.api.chat import chat_router
from app.api.conversation import conversation_router

app = FastAPI(
    title="Static Dynamic Continuous Batching",
    description="Learning AI inference and batching techniques",
    version="0.0.1",
)

app.include_router(chat_router)
app.include_router(conversation_router)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": "0.0.1",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )