import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional.
    pass


DATABASE_URL = os.getenv("DATABASE_URL")

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "llm")

BATCHING_MODE = os.getenv("BATCHING_MODE", "none")

STATIC_BATCH_SIZE = int(os.getenv("STATIC_BATCH_SIZE", "8"))

DYNAMIC_BATCH_SIZE = int(os.getenv("DYNAMIC_BATCH_SIZE", "32"))
DYNAMIC_TIMEOUT_MS = int(os.getenv("DYNAMIC_TIMEOUT_MS", "10"))

CONTINUOUS_MAX_ACTIVE_SEQUENCES = int(os.getenv("CONTINUOUS_MAX_ACTIVE_SEQUENCES", "32"))
