from typing import List, Optional

import openai
from openai.types import EmbeddingModel

from ragcite.core.interfaces.embedder import AsyncBaseEmbedder, BaseEmbedder
from ragcite.core.types import VectorLike


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        client: Optional[openai.OpenAI] = None,
        api_key: Optional[str] = None,
        model: EmbeddingModel = "text-embedding-3-small",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client or openai.OpenAI(api_key=self._api_key)

    def embed(self, text: str) -> VectorLike:
        response = self._client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding

    def batch_embed(self, texts: List[str]) -> List[VectorLike]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def batch_embed_dict(self, texts: dict[int, str]) -> dict[int, VectorLike]:
        response = self._client.embeddings.create(
            input=list(texts.values()), model=self._model
        )
        return {idx: item.embedding for idx, item in zip(texts.keys(), response.data)}

    def similarity(self, vec1: VectorLike, vec2: VectorLike) -> float:
        return super().similarity(vec1, vec2)


class AsyncOpenAIEmbedder(AsyncBaseEmbedder):
    def __init__(
        self,
        client: Optional[openai.AsyncOpenAI] = None,
        api_key: Optional[str] = None,
        model: EmbeddingModel = "text-embedding-3-small",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client or openai.AsyncOpenAI(api_key=self._api_key)

    async def embed(self, text: str) -> VectorLike:
        response = await self._client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding

    async def batch_embed(self, texts: List[str]) -> List[VectorLike]:
        response = await self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    async def batch_embed_dict(self, texts: dict[int, str]) -> dict[int, VectorLike]:
        response = await self._client.embeddings.create(
            input=list(texts.values()), model=self._model
        )
        return {idx: item.embedding for idx, item in zip(texts.keys(), response.data)}

    def similarity(self, vec1: VectorLike, vec2: VectorLike) -> float:
        return super().similarity(vec1, vec2)
