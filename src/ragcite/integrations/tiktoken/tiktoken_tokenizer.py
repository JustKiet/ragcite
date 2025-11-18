import tiktoken

from ragcite.core.interfaces import BaseTokenizer
from ragcite.core.interfaces.tokenizer import Token, TokenID


class TiktokenTokenizer(BaseTokenizer):
    def __init__(self, encoding: str):
        self._encoding = tiktoken.get_encoding(encoding)
        self._last_token_ids: list[int] | None = None  # Cache for detokenize

    @classmethod
    def from_model(cls, model_name: str) -> "TiktokenTokenizer":
        encoding = tiktoken.encoding_for_model(model_name)
        encoding_name = encoding.name
        return cls(encoding_name)

    def tokenize(self, text: str) -> list[Token]:
        token_ids = self._encoding.encode(text)
        # Cache the token IDs for potential use in detokenize
        self._last_token_ids = token_ids
        # Use decode_single_token_bytes to properly handle multi-byte UTF-8 characters
        tokens = [
            self._encoding.decode_single_token_bytes(token_id).decode(
                "utf-8", errors="replace"
            )
            for token_id in token_ids
        ]
        return tokens

    def detokenize(self, tokens: list[Token]) -> str:
        # If we have cached token IDs from the last tokenize call, use them
        # This is the only reliable way to reconstruct text with partial UTF-8 sequences
        if self._last_token_ids is not None and len(tokens) == len(
            self._last_token_ids
        ):
            result = self._encoding.decode(self._last_token_ids)
            self._last_token_ids = None  # Clear cache after use
            return result

        # Fallback: try to reconstruct by joining tokens
        # This works for most cases but may fail with partial UTF-8 sequences
        # Encode the combined string using surrogateescape to handle replacement characters
        return "".join(tokens)

    def encode(self, text: str) -> list[TokenID]:
        return self._encoding.encode(text)

    def decode(self, token_ids: list[TokenID]) -> str:
        return self._encoding.decode(token_ids)
