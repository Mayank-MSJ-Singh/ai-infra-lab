import asyncio
import logging

from app.batching.batch_item import BatchItem
from app.config import DYNAMIC_BATCH_SIZE, DYNAMIC_TIMEOUT_MS


logger = logging.getLogger(__name__)


class DynamicBatching:
    def __init__(self):
        # Maximum number of requests to collect before creating a batch.
        self.batch_size = DYNAMIC_BATCH_SIZE
        # Wait time for the batch to be processed.
        self.timeout = DYNAMIC_TIMEOUT_MS / 1000   # in seconds

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
            name="dynamic-batching-worker",
        )

        logger.info(
            "Dynamic batching initialized (batch_size=%d)",
            self.batch_size,
        )

    async def _worker(self):
        # Import here to avoid circular import:
        # llm_service → dynamic → llm_service
        from app.services.llm_service import chat

        logger.info("Dynamic batching worker started")

        while True:
            # Wait until enough requests are available to form
            # one complete Dynamic batch.
            batch = await self._collect_batch()

            logger.info(
                "Collected Dynamic batch with %d requests",
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
                    "Completed Dynamic batch with %d requests",
                    len(batch),
                )

            except Exception:
                # Never let the worker crash.
                #
                # If one batch fails, log the error and notify every waiting
                # request by completing its Future with an exception.
                logger.exception("Failed to process Dynamic batch")

                for batch_item in batch:
                    if not batch_item.future.done():
                        batch_item.future.set_exception(
                            RuntimeError("Dynamic batch processing failed")
                        )

    async def _collect_batch(self) -> list[BatchItem]:
        """
        Collect requests until either:
        1. The batch reaches the configured size.
        2. The timeout expires.
        """

        batch: list[BatchItem] = []

        # Wait indefinitely for the first request.
        first_item = await self.queue.get()
        batch.append(first_item)

        # Start the timeout once the first request arrives.
        deadline = (
            asyncio.get_running_loop().time()
            + self.timeout
        )

        while len(batch) < self.batch_size:
            remaining = deadline - asyncio.get_running_loop().time()

            if remaining <= 0:
                break

            try:
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=remaining,
                )

                batch.append(item)

            except asyncio.TimeoutError:
                break

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
            "Request added to Dynamic batching queue (queue_size=%d)",
            self.queue.qsize(),
        )

        # Wait until the worker completes this request.
        return await future