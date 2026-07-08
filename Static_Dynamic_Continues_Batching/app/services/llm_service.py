from openai import AsyncOpenAI

from app.config import LLM_BASE_URL


client = AsyncOpenAI(
    api_key="dummy",
    base_url=LLM_BASE_URL,
)


class Chat:
    async def get_response(self, messages: list[dict]) -> str:
        response = await client.chat.completions.create(
            model="llama3.2",
            messages=messages,
        )

        return response.choices[0].message.content or ""