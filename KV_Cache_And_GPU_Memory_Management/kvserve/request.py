from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class SequenceStatus(Enum):
    """
    Defines the complete state machine lifecycle for a single sequence/prompt.
    
    The Scheduler checks `sequence.status` on every engine step to make execution,
    offloading, and memory management decisions.

    Lifecycle Transitions:
        [WAITING] ---> [RUNNING] ---> [FINISHED_STOP] or [FINISHED_LENGTH_CAPPED]
                            |
                            v (under heavy memory pressure)
                        [SWAPPED]
    """

    WAITING = auto()
    """
    The request has been submitted to the engine's incoming queue.
    It is waiting for the Scheduler to allocate initial physical KV pages
    before running its prefill pass.
    """

    RUNNING = auto()
    """
    The request has allocated GPU pages in VRAM and is actively generating 
    tokens during decode forward passes.
    """

    SWAPPED = auto()
    """
    The sequence's KV cache pages were temporarily evicted to CPU host RAM 
    because VRAM hit high memory pressure. It will resume once VRAM frees up.
    """

    FINISHED_STOP = auto()
    """
    Generation completed normally. The model produced an End-Of-Sequence (EOS) 
    token or matched a specified stop token ID. VRAM pages can now be freed.
    """

    FINISHED_LENGTH_CAPPED = auto()
    """
    Generation halted because the sequence reached the requested max_tokens limit. 
    VRAM pages can now be freed.
    """


@dataclass
class SamplingParams:
    """
    Holds token sampling and generation parameters for a specific request.
    Dictates how logits are converted to token choices during the generation loop.
    """

    max_tokens: int = 50
    """
    The maximum number of new tokens allowed for generation.
    Once generated count == max_tokens, status transitions to FINISHED_LENGTH_CAPPED.
    """

    temperature: float = 0.0
    """
    Scales output logits prior to soft-max.
    0.0 forces deterministic Greedy Decoding (argmax).
    Values > 0.0 increase diversity/randomness.
    """

    top_p: float = 1.0
    """
    Nucleus sampling threshold.
    Filters the smallest cumulative probability tail out of the vocabulary distribution.
    1.0 means sample across all valid tokens in the vocabulary.
    """

    stop_token_ids: List[int] = field(default_factory=list)
    """
    Custom token IDs that trigger immediate generation halting when produced.
    Uses field(default_factory=list) to ensure every SamplingParams instance gets its
    own independent list in memory rather than sharing a mutable default across instances.
    """


@dataclass
class Request:
    """
    The immutable entry payload submitted to the engine by a client.
    Contains the raw prompt input tokenized into integer IDs alongside generation rules.
    """

    request_id: str
    """
    Unique identifier (e.g., 'req_001') assigned to the request. Used across system
    logs, sequence tracking, and for routing generated stream outputs to clients.
    """

    prompt_token_ids: List[int]
    """
    The input prompt text converted into integer token IDs by the tokenizer.
    Passed directly into the model's prefill phase.
    """

    params: SamplingParams
    """
    The specific generation rules (temperature, max_tokens, etc.) tied to this prompt.
    """


@dataclass
class RequestOutput:
    """
    The output payload streamed or returned back to the client after an engine step.
    Combines client-facing text with engine-facing raw token data.
    """

    request_id: str
    """
    Matches Request.request_id to map streamed tokens back to the caller.
    """

    new_text: str
    """
    Human-readable text snippet decoded from the latest generated token ID(s).
    Allows client applications to stream text directly without needing a local tokenizer.
    """

    output_token_ids: List[int]
    """
    Cumulative list of all integer token IDs generated for this request so far.
    Used by the engine to evaluate stop conditions and handle sub-word UTF-8 boundaries.
    """

    finished: bool
    """
    Flag indicating if generation for this request has concluded.
    When True, the client can close the stream connection.
    """

    finish_reason: Optional[str] = None
    """
    Explains why generation ended: 'stop' (EOS/stop token) or 'length' (hit max_tokens).
    Remains None while status is RUNNING or WAITING.
    """