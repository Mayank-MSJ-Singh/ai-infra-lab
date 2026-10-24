from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Message


class MessageService:

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> Message:

        db: Session = SessionLocal()

        try:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
            )

            db.add(message)
            db.commit()
            db.refresh(message)

            return message

        finally:
            db.close()

    def get_messages(self, conversation_id: str) -> list[dict]:

        db: Session = SessionLocal()

        try:
            messages = (
                db.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
                .all()
            )

            return [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ]

        finally:
            db.close()


message_service = MessageService()