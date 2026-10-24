from abc import ABC, abstractmethod


class BatchingStrategy(ABC):

    @abstractmethod
    async def get_response(self, messages: list[dict]) -> str:
        pass