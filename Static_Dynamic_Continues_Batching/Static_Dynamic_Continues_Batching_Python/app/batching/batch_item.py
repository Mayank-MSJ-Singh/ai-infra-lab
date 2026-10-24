from dataclasses import dataclass
import asyncio


@dataclass
class BatchItem:
    messages: list[dict]
    future: asyncio.Future[str]