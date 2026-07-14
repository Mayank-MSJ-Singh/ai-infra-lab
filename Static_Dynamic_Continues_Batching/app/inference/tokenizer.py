import logging

from transformers import AutoTokenizer

from app.config import MODEL_NAME


logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self):
        logger.info("Loading tokenizer: %s", MODEL_NAME)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
        )

        # Some causal LLMs don't define a pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Tokenizer loaded successfully")

    def encode(self, text: str):
        """
        Convert text into token IDs.
        """

        return self.tokenizer(
            text,
            return_tensors="pt",
        )

    def decode(self, token_ids):
        """
        Convert token IDs back into text.
        """

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )


tokenizer = Tokenizer()