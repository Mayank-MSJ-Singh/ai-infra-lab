from fastapi import APIRouter
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Conversation


conversation_router = APIRouter(
    prefix="/conversation",
    tags=["Conversation"],
)


@conversation_router.post("/")
async def create_conversation():
    db: Session = SessionLocal()

    try:
        conversation = Conversation()

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return {
            "conversation_id": conversation.id
        }

    finally:
        db.close()