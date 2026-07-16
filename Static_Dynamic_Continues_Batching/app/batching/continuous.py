import asyncio
import logging

from app.batching.sequence import Sequence
from app.config import CONTINUOUS_MAX_ACTIVE_SEQUENCES
from app.inference.engine import InferenceEngine


logger = logging.getLogger(__name__)

class ContinuousBatching:
    def __init__(self):
        # Maximum number of sequences that can be active
        # (running on the model) at the same time.
        self.max_batch_size = CONTINUOUS_MAX_ACTIVE_SEQUENCES

        # New requests wait here until the scheduler admits them.
        self.waiting_queue: asyncio.Queue[Sequence] = asyncio.Queue()

        # Requests currently being decoded.
        self.active_sequences: list[Sequence] = []

        # Initialize the inference engine
        self.inference_engine = InferenceEngine()

        # Start the scheduler.
        self.scheduler = asyncio.create_task(
            self._scheduler()
        )

        logger.info(
            "Continuous batching initialized (max_batch_size=%d)",
            self.max_batch_size,
        )

    async def submit(self, messages: list[dict]) -> str:
        # Future used by the scheduler to return the final response
        # back to the original request.
        future: asyncio.Future[str] = asyncio.Future()

        # Wrap the request into a Sequence.
        sequence = Sequence(
            messages=messages,
            future=future,
        )

        # Add the request to the waiting queue.
        #
        # The scheduler will later decide when this request should
        # become active.
        await self.waiting_queue.put(sequence)

        logger.debug(
            "Request added to waiting queue (queue_size=%d)",
            self.waiting_queue.qsize(),
        )

        # Suspend this coroutine until the scheduler finishes
        # generating the response.
        return await future

    async def _scheduler(self):
        logger.info("Continuous batching scheduler started")

        while True:
            if not self.active_sequences:
                # Suspend until a request arrives
                first_sequence = await self.waiting_queue.get()
                self.active_sequences.append(first_sequence)

            # Move waiting requests into the active batch
            # if there is available capacity.
            await self._admit_new_requests()

            # Generate one decoding step for every active sequence.
            await self._decode_step()

            # Remove completed sequences and wake any waiting
            # HTTP requests.
            self._remove_finished_sequences()

            # Yield control to the event loop to allow other tasks to run
            await asyncio.sleep(0.0)

    async def _admit_new_requests(self):
        """
        Move waiting requests into the active batch until
        the maximum number of active sequences is reached.
        """

        while (
            len(self.active_sequences) < self.max_batch_size
            and not self.waiting_queue.empty()
        ):
            sequence = self.waiting_queue.get_nowait()
            self.active_sequences.append(sequence)

            logger.debug(
                "Sequence admitted (active=%d, waiting=%d)",
                len(self.active_sequences),
                self.waiting_queue.qsize(),
            )

    async def _decode_step(self):
        """
        Advance every active sequence by one decoding step.

        The inference engine will later:
        - Generate one token for each active sequence.
        - Update the sequence state.
        - Mark completed sequences.
        """

        if not self.active_sequences:
            return

        await self.inference_engine.decode_step(
            self.active_sequences
        )
        logger.debug(
            "Decoded one step for %d active sequences",
            len(self.active_sequences),
        )

    def _remove_finished_sequences(self):
        """
        Remove completed sequences from the active list.

        Before removing them, wake the waiting HTTP request by
        completing its Future with the generated response.
        """

        remaining_sequences: list[Sequence] = []

        for sequence in self.active_sequences:
            if sequence.finished:
                # Wake the coroutine waiting inside submit().
                sequence.future.set_result(
                    sequence.generated_text
                )

                logger.debug(
                    "Sequence completed"
                )
            else:
                remaining_sequences.append(sequence)

        self.active_sequences = remaining_sequences

    