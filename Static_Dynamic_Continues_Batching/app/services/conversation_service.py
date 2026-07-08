from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Conversation


class ConversationService:

    def create_conversation(self) -> Conversation:
        db: Session = SessionLocal()

        try:
            conversation = Conversation()

            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            return conversation

        finally:
            db.close()

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        db: Session = SessionLocal()

        try:
            return (
                db.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first()
            )

        finally:
            db.close()

    def get_conversations(self) -> list[Conversation]:
        db: Session = SessionLocal()

        try:
            return (
                db.query(Conversation)
                .order_by(Conversation.created_at.desc())
                .all()
            )

        finally:
            db.close()

    def delete_conversation(self, conversation_id: str) -> bool:
        db: Session = SessionLocal()

        try:
            conversation = (
                db.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first()
            )

            if conversation is None:
                return False

            db.delete(conversation)
            db.commit()

            return True

        finally:
            db.close()


conversation_service = ConversationService()