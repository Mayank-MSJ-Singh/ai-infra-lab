import os
import unittest

# Configure the database to use a local test SQLite database file
os.environ["DATABASE_URL"] = "sqlite:///test.db"
os.environ["LLM_BASE_URL"] = "http://localhost:12434/v1"

from app.db import Base, engine
from app.services.conversation_service import conversation_service
from app.services.message_service import message_service

from sqlalchemy import event

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class TestServices(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create all tables in the SQLite in-memory database
        Base.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls):
        # Clean up tables and remove database file
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
        if os.path.exists("test.db"):
            os.remove("test.db")

    def setUp(self):
        # Clean up database records before each test to ensure isolation
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    def test_create_conversation(self):
        # Edge Case: Create a new conversation and check fields
        conversation = conversation_service.create_conversation()
        self.assertIsNotNone(conversation.id)
        self.assertIsNotNone(conversation.created_at)

        # Retrieve and verify it exists
        retrieved = conversation_service.get_conversation(conversation.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, conversation.id)

    def test_get_non_existent_conversation(self):
        # Edge Case: Try to retrieve a conversation with an invalid UUID
        retrieved = conversation_service.get_conversation("non-existent-uuid")
        self.assertIsNone(retrieved)

    def test_get_conversations_empty_and_multiple(self):
        # Edge Case: Check listing when no conversations exist
        self.assertEqual(conversation_service.get_conversations(), [])

        # Create multiple conversations
        c1 = conversation_service.create_conversation()
        c2 = conversation_service.create_conversation()

        conversations = conversation_service.get_conversations()
        self.assertEqual(len(conversations), 2)
        # Verify ordering (newest first, based on created_at desc)
        self.assertEqual(conversations[0].id, c2.id)
        self.assertEqual(conversations[1].id, c1.id)

    def test_delete_conversation(self):
        # Edge Case: Delete a valid conversation
        c = conversation_service.create_conversation()
        success = conversation_service.delete_conversation(c.id)
        self.assertTrue(success)
        self.assertIsNone(conversation_service.get_conversation(c.id))

        # Edge Case: Delete a non-existent conversation
        success_non_existent = conversation_service.delete_conversation("non-existent")
        self.assertFalse(success_non_existent)

    def test_add_message_success(self):
        # Edge Case: Add messages to an active conversation
        c = conversation_service.create_conversation()
        msg1 = message_service.add_message(c.id, "user", "Hello AI")
        self.assertEqual(msg1.conversation_id, c.id)
        self.assertEqual(msg1.role, "user")
        self.assertEqual(msg1.content, "Hello AI")

        # Verify messages are ordered correctly
        msg2 = message_service.add_message(c.id, "assistant", "Hello human")
        
        history = message_service.get_messages(c.id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"role": "user", "content": "Hello AI"})
        self.assertEqual(history[1], {"role": "assistant", "content": "Hello human"})

    def test_add_message_invalid_conversation(self):
        # Edge Case: Add a message to a non-existent conversation (should fail due to Foreign Key constraint)
        from sqlalchemy.exc import IntegrityError
        with self.assertRaises(IntegrityError):
            message_service.add_message("invalid-id", "user", "Hello")

    def test_cascade_delete_messages(self):
        # Edge Case: Deleting a conversation should cascade delete all of its messages
        c = conversation_service.create_conversation()
        message_service.add_message(c.id, "user", "Delete me")

        # Delete conversation
        conversation_service.delete_conversation(c.id)

        # Check that messages no longer exist
        messages = message_service.get_messages(c.id)
        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
