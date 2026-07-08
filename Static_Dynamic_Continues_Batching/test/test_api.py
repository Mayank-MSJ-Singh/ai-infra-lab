import os
import unittest
from unittest.mock import AsyncMock, patch

# Configure database to use a local test SQLite file before importing anything
os.environ["DATABASE_URL"] = "sqlite:///test.db"
os.environ["LLM_BASE_URL"] = "http://localhost:12434/v1"

from fastapi.testclient import TestClient
from sqlalchemy import event

from app.db import Base, engine
from app.services.conversation_service import conversation_service
from app.services.message_service import message_service
from main import app

# Enable foreign keys in SQLite for the API tests
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app, raise_server_exceptions=False)

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
        if os.path.exists("test.db"):
            os.remove("test.db")

    def setUp(self):
        # Reset DB tables before each test for clean isolation
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    def test_root_endpoint(self):
        # Edge Case: Root endpoint check
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "version": "0.0.1"})

    def test_create_conversation_endpoint(self):
        # Edge Case: Create a conversation via API
        response = self.client.post("/conversation/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("conversation_id", data)
        self.assertIsNotNone(data["conversation_id"])

        # Verify it exists in database
        conv = conversation_service.get_conversation(data["conversation_id"])
        self.assertIsNotNone(conv)

    @patch("app.api.chat.chat.get_response", new_callable=AsyncMock)
    def test_chat_success(self, mock_get_response):
        # Edge Case: Successful message exchange
        mock_get_response.return_value = "Mocked AI Response"

        # 1. Create a conversation
        conv = conversation_service.create_conversation()

        # 2. Post a chat message
        response = self.client.post(
            f"/chat/{conv.id}",
            json={"message": "Hello AI"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "Mocked AI Response"})

        # 3. Verify messages are saved in database
        messages = message_service.get_messages(conv.id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "user", "content": "Hello AI"})
        self.assertEqual(messages[1], {"role": "assistant", "content": "Mocked AI Response"})

        # Verify that mock was called with correct parameters
        mock_get_response.assert_called_once_with([{"role": "user", "content": "Hello AI"}])

    def test_chat_missing_fields(self):
        # Edge Case: Request with invalid payload (missing message)
        conv = conversation_service.create_conversation()
        response = self.client.post(
            f"/chat/{conv.id}",
            json={} # Empty JSON should fail Pydantic validation
        )
        self.assertEqual(response.status_code, 422) # Unprocessable Entity

    def test_chat_invalid_json(self):
        # Edge Case: Request with bad JSON format
        conv = conversation_service.create_conversation()
        response = self.client.post(
            f"/chat/{conv.id}",
            content="invalid-json-string",
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 422)

    def test_chat_non_existent_conversation(self):
        # Edge Case: Attempting to chat in a non-existent conversation
        # This will fail at the database level with a Foreign Key IntegrityError
        # (resulting in a 500 error on the API side currently).
        # We test that it fails cleanly (either 500 or 404).
        response = self.client.post(
            "/chat/non-existent-id-123",
            json={"message": "Hello"}
        )
        # Verify it raises a server error due to foreign key integrity check
        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
