from openai import AsyncOpenAI

from app.batching.base import BatchingStrategy
from app.config import LLM_BASE_URL, MODEL_NAME


client = AsyncOpenAI(
    api_key="dummy",
    base_url=LLM_BASE_URL,
)


class NoBatching(BatchingStrategy):

    async def get_response(self, messages: list[dict]) -> str:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )

        return response.choices[0].message.content or ""