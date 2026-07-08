import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")