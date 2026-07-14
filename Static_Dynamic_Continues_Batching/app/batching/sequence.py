from dataclasses import dataclass

from app.batching.batch_item import BatchItem

@dataclass
class Sequence(BatchItem):
    generated_text: str = ""
    finished: bool = False