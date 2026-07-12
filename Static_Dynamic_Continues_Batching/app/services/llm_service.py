import asyncio
import logging

from openai import AsyncOpenAI

from app.batching.batch_item import BatchItem
from app.batching.dynamic import DynamicBatching
from app.batching.static import StaticBatching
from app.config import (
    BATCHING_MODE,
    LLM_BASE_URL,
    MODEL_NAME,
)


logger = logging.getLogger(__name__)


client = AsyncOpenAI(
    api_key="dummy",
    base_url=LLM_BASE_URL,
)


class Chat:
    def __init__(self):
        # Lazily initialize batching strategies.
        #
        # Only the configured strategy will create its background worker.
        self.static_batching = None
        self.dynamic_batching = None

    async def get_response(self, messages: list[dict]) -> str:
        """
        Entry point used by the application.

        Depending on the configured batching mode, the request is either
        executed immediately or forwarded to the configured batching
        strategy.
        """

        if BATCHING_MODE == "none":
            return await self.generate(messages)

        if BATCHING_MODE == "static":
            if self.static_batching is None:
                self.static_batching = StaticBatching()

            return await self.static_batching.submit(messages)

        if BATCHING_MODE == "dynamic":
            if self.dynamic_batching is None:
                self.dynamic_batching = DynamicBatching()

            return await self.dynamic_batching.submit(messages)

        raise ValueError(
            f"Unsupported batching mode: {BATCHING_MODE}"
        )

    async def generate(self, messages: list[dict]) -> str:
        """
        Execute a single inference request.
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
        Execute a batch of inference requests concurrently.

        Note:
        This is scheduler-side batching, not true backend batching.
        Every request is executed concurrently using asyncio.gather().

        If the serving backend later exposes a native batch API,
        only this method needs to change.
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