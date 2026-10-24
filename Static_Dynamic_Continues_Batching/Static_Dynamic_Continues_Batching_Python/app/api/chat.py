from fastapi import APIRouter
from pydantic import BaseModel

from app.services.llm_service import Chat
from app.services.message_service import message_service


class ChatRequest(BaseModel):
    message: str


chat_router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

chat = Chat()


@chat_router.post("/{conversation_id}")
async def chat_endpoint(
    conversation_id: str,
    chat_request: ChatRequest,
):
    # Save user message
    message_service.add_message(
        conversation_id,
        "user",
        chat_request.message,
    )

    # Load conversation history
    messages = message_service.get_messages(conversation_id)

    # Get LLM response
    response = await chat.get_response(messages)

    # Save assistant response
    message_service.add_message(
        conversation_id,
        "assistant",
        response,
    )

    return {
        "response": response
    }