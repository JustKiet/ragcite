from abc import ABC, abstractmethod
from typing import List

from ragcite.core.types import VectorLike
from ragcite.core.utils.similarity_algorithms import cosine_similarity


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> VectorLike:
        """
        Embed a single text into a vector representation.

        :param str text: The text to embed.
        """
        pass

    @abstractmethod
    def batch_embed(self, texts: List[str]) -> List[VectorLike]:
        pass

    @abstractmethod
    def batch_embed_dict(self, texts: dict[int, str]) -> dict[int, VectorLike]:
        pass

    def similarity(self, vec1: VectorLike, vec2: VectorLike) -> float:
        return cosine_similarity(vec1, vec2)


class AsyncBaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> VectorLike:
        """
        Asynchronously embed a single text into a vector representation.

        :param str text: The text to embed.
        """
        pass

    @abstractmethod
    async def batch_embed(self, texts: List[str]) -> List[VectorLike]:
        pass

    @abstractmethod
    async def batch_embed_dict(self, texts: dict[int, str]) -> dict[int, VectorLike]:
        pass

    def similarity(self, vec1: VectorLike, vec2: VectorLike) -> float:
        return cosine_similarity(vec1, vec2)
