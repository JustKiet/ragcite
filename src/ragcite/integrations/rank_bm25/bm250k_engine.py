from typing import Any, cast

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore

from ragcite.core.interfaces import BaseBM25Engine


class BM25OkEngine(BaseBM25Engine):
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
        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(query_tokens)  # type: ignore
        scores = cast(np.ndarray[Any, Any], scores)

        # Convert numpy array to list
        return [float(s) for s in scores]
