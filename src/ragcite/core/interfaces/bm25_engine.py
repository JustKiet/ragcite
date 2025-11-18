from abc import ABC, abstractmethod


class BaseBM25Engine(ABC):
    @abstractmethod
    def get_scores(
        self, query_tokens: list[str], tokenized_chunks: list[list[str]]
    ) -> list[float]:
        """
        Compute BM25 scores for lexical matching

        :param list[str] query_tokens: Tokenized query.
        :param list[list[str]] tokenized_chunks: List of tokenized context chunks.
        :return: List of BM25 scores corresponding to each context chunk.
        :rtype: list[float]
        """
        pass
