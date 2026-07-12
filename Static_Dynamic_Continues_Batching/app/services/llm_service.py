import asyncio
import logging

from openai import AsyncOpenAI

from app.batching.batch_item import BatchItem
from app.config import LLM_BASE_URL, MODEL_NAME


logger = logging.getLogger(__name__)


client = AsyncOpenAI(
    api_key="dummy",
    base_url=LLM_BASE_URL,
)


class Chat:
    async def generate(self, messages: list[dict]) -> str:
        """
        Generate a response for a single request.
        """

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )

        return response.choices[0].message.content or ""

    async def generate_batch(
        self,
        batch: list[BatchItem],
    ) -> list[str]:
        """
        Execute every request in the batch concurrently.

        Note:
        This is NOT true model-side batching. We simply send multiple
        requests concurrently using asyncio.gather().

        If the backend later supports real batching (vLLM, TensorRT-LLM,
        SGLang, etc.), only this function needs to change.
        """

        logger.info(
            "Executing inference for batch of %d requests",
            len(batch),
        )

        tasks = [
            self.generate(item.messages)
            for item in batch
        ]

        responses = await asyncio.gather(*tasks)

        logger.info(
            "Completed inference for batch of %d requests",
            len(batch),
        )

        return responses


chat = Chat()