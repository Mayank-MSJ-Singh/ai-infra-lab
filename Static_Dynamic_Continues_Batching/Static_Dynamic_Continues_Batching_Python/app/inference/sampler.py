import torch


class Sampler:
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Select the token with the highest probability.

        This is called Greedy Decoding.
        """

        return torch.argmax(
            logits,
            dim=-1,
        )


sampler = Sampler()