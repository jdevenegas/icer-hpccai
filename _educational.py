"""Educational implementation of byte pair encoding (BPE).

Based on OpenAI's tiktoken/_educational.py, with:
- user-supplied training text via train_simple_encoding(data, ...)
- live, updating training visualization for Jupyter notebooks
- configurable delay between merges
- graceful stopping when no more byte pairs can be merged
"""

from __future__ import annotations

import collections
import time

import regex
import tiktoken


class SimpleBytePairEncoding:
    def __init__(
        self, *, pat_str: str, mergeable_ranks: dict[bytes, int]
    ) -> None:
        """Create a simple educational BPE encoding."""
        self.pat_str = pat_str
        self.mergeable_ranks = mergeable_ranks
        self._decoder = {
            token: token_bytes for token_bytes, token in mergeable_ranks.items()
        }
        self._pat = regex.compile(pat_str)

    def encode(
        self, text: str, visualise: str | None = None
    ) -> list[int]:
        """Encode text into token IDs."""
        words = self._pat.findall(text)
        tokens = []

        for word in words:
            word_bytes = word.encode("utf-8")
            word_tokens = bpe_encode(
                self.mergeable_ranks, word_bytes, visualise=visualise
            )
            tokens.extend(word_tokens)

        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decode token IDs into bytes."""
        return b"".join(self._decoder[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs into a string."""
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        """Return the byte sequence corresponding to each token ID."""
        return [self._decoder[token] for token in tokens]

    @staticmethod
    def train(
        training_data: str,
        vocab_size: int,
        pat_str: str,
        visualise: str | None = "standard",
        delay: float = 0.1,
    ) -> "SimpleBytePairEncoding":
        """Train a BPE tokenizer on supplied text."""
        mergeable_ranks = bpe_train(
            data=training_data,
            vocab_size=vocab_size,
            pat_str=pat_str,
            visualise=visualise,
            delay=delay,
        )
        return SimpleBytePairEncoding(
            pat_str=pat_str, mergeable_ranks=mergeable_ranks
        )

    @staticmethod
    def from_tiktoken(encoding):
        """Create an educational encoding from an existing tiktoken encoding."""
        if isinstance(encoding, str):
            encoding = tiktoken.get_encoding(encoding)

        return SimpleBytePairEncoding(
            pat_str=encoding._pat_str,
            mergeable_ranks=encoding._mergeable_ranks,
        )


def _clear_live_output() -> None:
    """Clear the previous output in Jupyter, or clear a terminal screen."""
    try:
        from IPython.display import clear_output

        clear_output(wait=True)
    except ImportError:
        print("\033[2J\033[H", end="")


def bpe_encode(
    mergeable_ranks: dict[bytes, int],
    input: bytes,
    visualise: str | None = None,
) -> list[int]:
    """Apply learned BPE merges to one byte sequence."""
    parts = [bytes([b]) for b in input]

    while True:
        if visualise:
            if visualise in ("colour", "color"):
                visualise_tokens(parts)
            elif visualise == "simple":
                print(parts)

        min_idx = None
        min_rank = None

        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        if min_rank is None:
            break

        assert min_idx is not None
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2:]
        )

    if visualise:
        print()

    return [mergeable_ranks[part] for part in parts]


def bpe_train(
    data: str,
    vocab_size: int,
    pat_str: str,
    visualise: str | None = "standard",
    delay: float = 0.1,
) -> dict[bytes, int]:
    """Train BPE merges from text, optionally showing each merge live."""
    if vocab_size < 2**8:
        raise ValueError(
            "vocab_size must be at least 256, so we can encode all bytes"
        )
    if delay < 0:
        raise ValueError("delay must be non-negative")

    # The initial vocabulary contains all 256 possible byte values.
    ranks: dict[bytes, int] = {
        bytes([i]): i for i in range(2**8)
    }

    # Split the corpus into regex-defined pieces, then into individual bytes.
    words: list[list[bytes]] = [
        [bytes([b]) for b in word.encode("utf-8")]
        for word in regex.findall(pat_str, data)
    ]

    if not words:
        raise ValueError("The training data is empty or produced no regex matches.")

    while len(ranks) < vocab_size:
        stats = collections.Counter()

        for piece in words:
            for pair in zip(piece[:-1], piece[1:]):
                stats[pair] += 1

        # A small corpus may run out of mergeable pairs before vocab_size.
        if not stats:
            if visualise == "live":
                _clear_live_output()
            if visualise:
                print("Training stopped: no more adjacent byte pairs to merge.")
                print(f"Final vocabulary size: {len(ranks)}")
            break

        most_common_pair = max(stats, key=lambda pair: stats[pair])
        token_bytes = most_common_pair[0] + most_common_pair[1]
        token = len(ranks)
        ranks[token_bytes] = token

        # Apply this merge everywhere in the training corpus.
        new_words = []

        for word in words:
            new_word = []
            i = 0

            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_common_pair:
                    new_word.append(token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            if i == len(word) - 1:
                new_word.append(word[i])

            new_words.append(new_word)

        words = new_words

        # Standard mode prints every merge; live mode replaces the prior display.
        if visualise:
            if visualise == "live":
                _clear_live_output()

            if visualise not in ("standard", "live", "simple"):
                raise ValueError(
                    "visualise must be 'standard', 'live', 'simple', or None"
                )

            print(f"The current most common pair is {most_common_pair[0]} + {most_common_pair[1]}")
            print(f"So we made {token_bytes} our {len(ranks)}th token")
            print("Now the first fifty words in our training data look like:\\n")

            preview = [token for word in words[:50] for token in word]
            if visualise in ("standard", "live"):
                visualise_tokens(preview)
            elif visualise == "simple":
                for word in words[:20]:
                    print(word)

            if visualise == "live" and delay:
                time.sleep(delay)

    if visualise == "live":
        print("\\nTraining complete.")
        print(f"Final vocabulary size: {len(ranks)}")

    return ranks


def visualise_tokens(token_values: list[bytes]) -> None:
    """Print tokens with alternating background colors."""
    background = [
        f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]
    ]

    unicode_token_values = [
        token.decode("utf-8", errors="replace") for token in token_values
    ]

    running_length = 0
    last_color = None

    for token in unicode_token_values:
        color = background[running_length % len(background)]

        if color == last_color:
            color = background[(running_length + 1) % len(background)]

        last_color = color
        running_length += len(token)
        print(color + token, end="")

    print("\u001b[0m")


def train_simple_encoding(
    data: str,
    vocab_size: int = 600,
    visualise: str | None = "live",
    delay: float = 0.1,
) -> SimpleBytePairEncoding:
    """Train an educational BPE tokenizer on user-supplied text.

    Args:
        data: Training corpus as a Python string.
        vocab_size: Target vocabulary size, including the initial 256 bytes.
        visualise: "standard" (print every merge), "live" (update one
            Jupyter output), "simple" (print byte-token lists), or None.
        delay: Seconds to pause after each merge in live mode.
    """
    gpt2_pattern = (
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| """
        r"""[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    return SimpleBytePairEncoding.train(
        training_data=data,
        vocab_size=vocab_size,
        pat_str=gpt2_pattern,
        visualise=visualise,
        delay=delay,
    )
