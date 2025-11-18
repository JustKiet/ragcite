from abc import ABC, abstractmethod
from typing import TypeAlias

Token: TypeAlias = str
TokenID: TypeAlias = int


class BaseTokenizer(ABC):
    def __init__(self, tokenizer_name: str) -> None:
        super().__init__()
        self._tokenizer_name: str = tokenizer_name

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @abstractmethod
    def tokenize(self, text: str) -> list[Token]:
        """
        Tokenize the given text into a list of tokens.

        :param text: The text to tokenize.
        :return: A list of tokens.
        """
        pass

    @abstractmethod
    def detokenize(self, tokens: list[Token]) -> str:
        """
        Detokenize the given list of tokens back into text.

        :param tokens: The list of tokens to detokenize.
        :return: The detokenized text.
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> list[TokenID]:
        """
        Encode the given text into a list of token IDs.

        :param text: The text to encode.
        :return: A list of token IDs.
        """
        pass

    @abstractmethod
    def decode(self, token_ids: list[TokenID]) -> str:
        """
        Decode the given list of token IDs back into text.

        :param token_ids: The list of token IDs to decode.
        :return: The decoded text.
        """
        pass
