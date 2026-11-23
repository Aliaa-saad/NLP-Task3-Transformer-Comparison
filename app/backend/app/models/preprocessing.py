"""Tokenization helpers that reproduce each model's training-time preprocessing exactly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def head_tail_truncate(
    token_ids: list[int],
    max_len: int = 256,
    head_tokens: int = 192,
    tail_tokens: int = 64,
) -> list[int]:
    """Keep leading head_tokens and trailing tail_tokens (preserving [CLS] and [SEP]).

    Copied from the Model A notebook (src/data.py) so inference matches training.
    """
    if len(token_ids) <= max_len:
        return list(token_ids)
    head = token_ids[:head_tokens] if head_tokens > 0 else []
    # Guard tail_tokens == 0 because Python slices [-0:] as [0:].
    tail = token_ids[-tail_tokens:] if tail_tokens > 0 else []
    return (head + tail)[:max_len]


@dataclass
class EncodedText:
    input_ids: list[int]
    original_tokens: int

    @property
    def num_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def truncated(self) -> bool:
        return self.original_tokens > len(self.input_ids)


def encode_model_a(
    tokenizer: Any,
    text: str,
    max_len: int = 256,
    head_tokens: int = 192,
    tail_tokens: int = 64,
) -> EncodedText:
    """bert-base-uncased ids with special tokens, no padding, then head+tail truncation."""
    # verbose=False silences the ">512 tokens" warning; we truncate head+tail ourselves.
    ids = tokenizer(text, add_special_tokens=True, truncation=False, padding=False, verbose=False)["input_ids"]
    kept = head_tail_truncate(ids, max_len=max_len, head_tokens=head_tokens, tail_tokens=tail_tokens)
    return EncodedText(input_ids=kept, original_tokens=len(ids))


def pad_batch(sequences: list[list[int]], pad_id: int = 0) -> list[list[int]]:
    """Right-pad a batch of id lists to the longest sequence."""
    longest = max((len(s) for s in sequences), default=0)
    return [s + [pad_id] * (longest - len(s)) for s in sequences]
