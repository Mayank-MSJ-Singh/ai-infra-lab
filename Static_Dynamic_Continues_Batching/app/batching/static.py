import asyncio
import logging

from app.batching.batch_item import BatchItem
from app.config import STATIC_BATCH_SIZE

from app.services.llm_service import chat


logger = logging.getLogger(__name__)


class StaticBatching:
    def __init__(self):
        # Maximum number of requests to collect before creating a batch.
        self.batch_size = STATIC_BATCH_SIZE

        # Shared queue where incoming requests wait until they are
        # picked up by the background worker.
        self.queue: asyncio.Queue[BatchItem] = asyncio.Queue()

        # Start the batching worker in the background.
        #
        # The worker runs for the lifetime of the application and
        # continuously waits for new requests to arrive in the queue.
        #
        # asyncio.create_task() ensures the worker runs concurrently
        # without blocking the rest of the application.
        self.worker = asyncio.create_task(
            self._worker(),
            name="static-batching-worker",
        )

        logger.info(
            "Static batching initialized (batch_size=%d)",
            self.batch_size,
        )

    async def _worker(self):
        logger.info("Static batching worker started")

        while True:
            # Wait until enough requests are available to form
            # one complete static batch.
            batch = await self._collect_batch()

            logger.info(
                "Collected static batch with %d requests",
                len(batch),
            )

            try:
                # Send the complete batch to the LLM service for inference.
                #
                # Note:
                # The batching layer only schedules requests.
                # The LLM service is responsible for executing them.
                responses = await chat.generate_batch(batch)

                # Match every response with the request that created it.
                #
                # Each request owns a Future. Setting the result wakes up
                # the coroutine waiting inside submit().
                for batch_item, response in zip(batch, responses):
                    batch_item.future.set_result(response)

                logger.info(
                    "Completed static batch with %d requests",
                    len(batch),
                )

            except Exception:
                # Never let the worker crash.
                #
                # If one batch fails, log the error and notify every waiting
                # request by completing its Future with an exception.
                logger.exception("Failed to process static batch")

                for batch_item in batch:
                    if not batch_item.future.done():
                        batch_item.future.set_exception(
                            RuntimeError("Static batch processing failed")
                        )

    async def _collect_batch(self) -> list[BatchItem]:
        """
        Collect exactly `batch_size` requests from the queue.

        The worker sleeps while waiting for requests.
        Once enough requests arrive, the batch is returned.
        """

        batch: list[BatchItem] = []

        while len(batch) < self.batch_size:
            item = await self.queue.get()
            batch.append(item)

        return batch

    async def submit(self, messages: list[dict]) -> str:
        # Future used by the worker to return the response back to
        # the original request.
        future: asyncio.Future[str] = asyncio.Future()
        
        # Wrap the request together with its Future.
        item = BatchItem(
            messages=messages,
            future=future,
        )

        # Add the request to the batching queue.
        await self.queue.put(item)

        logger.debug(
            "Request added to static batching queue (queue_size=%d)",
            self.queue.qsize(),
        )

        # Wait until the worker completes this request.
        return await future