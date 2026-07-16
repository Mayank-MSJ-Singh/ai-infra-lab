import asyncio
import logging

from app.batching.batch_item import BatchItem
from app.batching.dynamic import DynamicBatching
from app.batching.static import StaticBatching
from app.batching.continuous import ContinuousBatching
from app.config import BATCHING_MODE


import torch

from app.inference.model import model
from app.inference.tokenizer import tokenizer
from app.inference.sampler import sampler


logger = logging.getLogger(__name__)


class Chat:
    def __init__(self):
        # Lazily initialize batching strategies.
        #
        # Only the configured strategy will create its background worker.
        self.static_batching = None
        self.dynamic_batching = None
        self.continuous_batching = None

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
        
        if BATCHING_MODE == "continuous":
            if self.continuous_batching is None:
                self.continuous_batching = ContinuousBatching()

            return await self.continuous_batching.submit(messages)

        raise ValueError(
            f"Unsupported batching mode: {BATCHING_MODE}"
        )
    
    @torch.inference_mode()
    async def generate(
        self,
        messages: list[dict],
        ) -> str:
        """
        Generate a response using the local Hugging Face model.
        """

        generated_tokens = []

        inputs = tokenizer.encode(messages)

        inputs = {
            key: value.to(model.model.device)
            for key, value in inputs.items()
        }

        for _ in range(128):
            # Run one forward pass.
            outputs = model.model(**inputs)

            # Predict the next token.
            next_token = sampler.sample(
                outputs.logits[:, -1, :]
            )

            # Stop if the model generated the EOS token.
            if next_token.item() == tokenizer.tokenizer.eos_token_id:
                break

            # Save the generated token.
            generated_tokens.append(next_token)

            # Feed the generated token back into the model.
            inputs["input_ids"] = torch.cat(
                [inputs["input_ids"], next_token.unsqueeze(0)],
                dim=1,
            )

            # Update the attention mask.
            attention = torch.ones(
                (1, 1),
                device=model.model.device,
                dtype=inputs["attention_mask"].dtype,
            )

            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], attention],
                dim=1,
            )

        generated = torch.cat(
            generated_tokens,
            dim=0,
        )

        return tokenizer.decode(generated)


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