import logging

import torch

from app.inference.model import model
from app.inference.tokenizer import tokenizer
from app.inference.sampler import sampler
from app.config import MAX_NEW_TOKENS

from app.batching.sequence import Sequence


logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Batched token-by-token inference engine for continuous batching.

    Each call to decode_step() advances every active sequence by
    exactly one token. This is the core difference from static/dynamic
    batching where generate() runs the full autoregressive loop per
    request.

    Lifecycle of a sequence through this engine:
    1. Prefill  — First call: tokenize messages into input_ids.
    2. Decode   — Each call: run one forward pass, sample one token.
    3. Finish   — When EOS or max tokens: mark sequence as done.
    """

    @torch.inference_mode()
    async def decode_step(self, active_sequences: list[Sequence]):
        """
        Advance every active sequence by one decoding step.

        Steps:
        1. Prefill any new sequences (tokenize their messages).
        2. Left-pad all sequences to the same length for batching.
        3. Run one batched forward pass through the model.
        4. Sample the next token for each sequence.
        5. Update each sequence's state (tokens, inputs, finished flag).
        """

        if not active_sequences:
            return

        # ── Step 1: Prefill ──────────────────────────────────────
        #
        # New sequences arrive with input_ids=None.
        # Tokenize their chat messages into model inputs.
        for seq in active_sequences:
            if seq.input_ids is None:
                inputs = tokenizer.encode(seq.messages)
                seq.input_ids = inputs["input_ids"].to(model.model.device)
                seq.attention_mask = inputs["attention_mask"].to(model.model.device)

        # ── Step 2: Left-pad to uniform length ───────────────────
        #
        # Sequences have different lengths (different prompts,
        # different number of tokens generated so far).
        #
        # To batch them into a single tensor, we left-pad shorter
        # sequences so that the LAST token in every row is the
        # most recent one.
        #
        # Left-padding is used (not right-padding) because causal
        # LMs predict the next token from the rightmost position.
        max_len = max(seq.input_ids.size(1) for seq in active_sequences)

        padded_input_ids = []
        padded_attention_masks = []

        for seq in active_sequences:
            seq_len = seq.input_ids.size(1)
            pad_len = max_len - seq_len

            if pad_len > 0:
                # Pad input_ids with the pad token on the LEFT.
                pad_ids = torch.full(
                    (1, pad_len),
                    tokenizer.tokenizer.pad_token_id,
                    dtype=seq.input_ids.dtype,
                    device=model.model.device,
                )
                padded_input_ids.append(
                    torch.cat([pad_ids, seq.input_ids], dim=1)
                )

                # Pad attention_mask with 0 on the LEFT.
                # 0 tells the model to ignore these padding positions.
                pad_mask = torch.zeros(
                    (1, pad_len),
                    dtype=seq.attention_mask.dtype,
                    device=model.model.device,
                )
                padded_attention_masks.append(
                    torch.cat([pad_mask, seq.attention_mask], dim=1)
                )
            else:
                padded_input_ids.append(seq.input_ids)
                padded_attention_masks.append(seq.attention_mask)

        # Stack all sequences into batch tensors.
        # Shape: (batch_size, max_len)
        batched_input_ids = torch.cat(padded_input_ids, dim=0)
        batched_attention_mask = torch.cat(padded_attention_masks, dim=0)

        # ── Step 3: Forward pass ─────────────────────────────────
        #
        # Run the model once for the entire batch.
        # This is the key efficiency gain of continuous batching:
        # one GPU kernel launch serves all active sequences.
        outputs = model.model(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
        )

        # ── Step 4: Sample next token ────────────────────────────
        #
        # Take logits from the LAST position of each sequence
        # (index -1, since we left-padded).
        next_tokens = sampler.sample(outputs.logits[:, -1, :])

        # ── Step 5: Update each sequence's state ─────────────────
        for i, seq in enumerate(active_sequences):
            next_token_id = next_tokens[i].item()

            # Record the generated token.
            seq.generated_tokens.append(next_token_id)

            # Append the new token to the sequence's own input_ids
            # (NOT the padded batch tensor — each sequence tracks
            # its own unpadded state).
            next_token_tensor = next_tokens[i].unsqueeze(0).unsqueeze(0)
            seq.input_ids = torch.cat(
                [seq.input_ids, next_token_tensor], dim=1
            )

            # Extend the attention mask by one position (value=1).
            attention_one = torch.ones(
                (1, 1),
                dtype=seq.attention_mask.dtype,
                device=model.model.device,
            )
            seq.attention_mask = torch.cat(
                [seq.attention_mask, attention_one], dim=1
            )

            # Check if this sequence is finished:
            #   - Generated the EOS token, or
            #   - Hit the maximum generation length.
            if (
                next_token_id == tokenizer.tokenizer.eos_token_id
                or len(seq.generated_tokens) >= MAX_NEW_TOKENS
            ):
                seq.finished = True
                seq.generated_text = tokenizer.decode(
                    seq.generated_tokens
                )

                logger.debug(
                    "Sequence finished (tokens=%d)",
                    len(seq.generated_tokens),
                )
