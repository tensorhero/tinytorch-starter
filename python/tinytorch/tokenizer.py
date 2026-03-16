"""Tokenizers — convert text to integer token ID sequences.

Contains:
- CharTokenizer: one character = one token.
- BPETokenizer: Byte Pair Encoding, merges frequent adjacent pairs.
"""

from __future__ import annotations


class CharTokenizer:
    """Character-level tokenizer: each unique character maps to one token ID.

    IDs are assigned in sorted character order.
    """

    def __init__(self) -> None:
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    # ================================================================
    # E11 — Tokenization
    # ================================================================

    def build_vocab(self, text: str) -> None:
        """Builds vocabulary from text. Sorted character order."""
        raise NotImplementedError("TODO: E11")

    def vocab_size(self) -> int:
        """Returns the vocabulary size."""
        raise NotImplementedError("TODO: E11")

    def encode(self, text: str) -> list[int]:
        """Encodes text into token IDs."""
        raise NotImplementedError("TODO: E11")

    def decode(self, ids: list[int]) -> str:
        """Decodes token IDs back to text."""
        raise NotImplementedError("TODO: E11")


class BPETokenizer:
    """BPE tokenizer: starts from character-level vocab and iteratively
    merges the most frequent adjacent token pairs.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []

    # ================================================================
    # E11 — Tokenization
    # ================================================================

    def train(self, text: str, vocab_size: int) -> None:
        """Trains BPE on text until reaching target vocab_size."""
        raise NotImplementedError("TODO: E11")

    def vocab_size(self) -> int:
        """Returns current vocabulary size."""
        raise NotImplementedError("TODO: E11")

    def encode(self, text: str) -> list[int]:
        """Encodes text using trained merge rules."""
        raise NotImplementedError("TODO: E11")

    def decode(self, ids: list[int]) -> str:
        """Decodes token IDs back to text."""
        raise NotImplementedError("TODO: E11")
