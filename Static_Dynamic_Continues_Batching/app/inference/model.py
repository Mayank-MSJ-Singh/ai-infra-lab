import logging

import torch
from transformers import AutoModelForCausalLM

from app.config import MODEL_NAME


logger = logging.getLogger(__name__)


class Model:
    def __init__(self):
        logger.info("Loading model: %s", MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.model.eval()

        logger.info("Model loaded successfully")


model = Model()