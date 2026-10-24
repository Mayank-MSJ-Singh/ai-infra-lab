from dataclasses import dataclass, field

import torch

from app.batching.batch_item import BatchItem


@dataclass
class Sequence(BatchItem):
    # Model inputs for the current decoding step.
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    # Tokens generated so far.
    generated_tokens: list[int] = field(default_factory=list)

    # Final decoded response.
    generated_text: str = ""

    # Indicates whether generation has completed.
    finished: bool = False