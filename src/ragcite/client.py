import warnings
from typing import Optional

from ragcite.core.interfaces import (
    BaseBM25Engine,
    BaseEmbedder,
    BaseLanguageDetector,
    BaseTokenizer,
)
from ragcite.core.utils.elbow_method import elbow_method
from ragcite.integrations.langdetect import LangDetectLanguageDetector
from ragcite.integrations.openai.embedder import OpenAIEmbedder
from ragcite.integrations.rank_bm25.bm250k_engine import BM25OkEngine
from ragcite.integrations.tiktoken.tiktoken_tokenizer import TiktokenTokenizer


class Ragcite:
    _language_detector: BaseLanguageDetector
    _embedder: Optional[BaseEmbedder]
    _bm25_engine: BaseBM25Engine
    _tokenizer: BaseTokenizer
    enable_same_language_semantics: bool
    min_absolute_score: float

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        language_detector: Optional[BaseLanguageDetector] = None,
        bm25_engine: Optional[BaseBM25Engine] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        enable_embedding: bool = True,
        enable_same_language_semantics: bool = False,
        min_absolute_score: float = 0.21,
    ) -> None:
        if not embedder and enable_embedding:
            self._embedder = OpenAIEmbedder()
        else:
            self._embedder = None

        if not language_detector:
            self._language_detector = LangDetectLanguageDetector()

        if not bm25_engine:
            self._bm25_engine = BM25OkEngine()

        if not tokenizer:
            self._tokenizer = TiktokenTokenizer(encoding="cl100k_base")

        # Initialize enable_same_language_semantics
        self.enable_same_language_semantics = enable_same_language_semantics

        if enable_same_language_semantics and not enable_embedding:
            warnings.warn(
                "enable_same_language_semantics is set to True but enable_embedding is False. "
                "Setting enable_same_language_semantics to False."
            )
            self.enable_same_language_semantics = False

        if enable_same_language_semantics and not self._embedder:
            warnings.warn(
                "enable_same_language_semantics is set to True but no embedder is provided. "
                "Setting enable_same_language_semantics to False."
            )
            self.enable_same_language_semantics = False

        self.min_absolute_score = min_absolute_score

    def _detect_cross_lingual(
        self, answer: str, context_chunks: dict[int, str]
    ) -> bool:
        """
        Detect if the answer and context chunks are in different languages.

        :param str answer: The generated answer text.
        :param dict[int, str] context_chunks: A dictionary mapping chunk IDs to their text.
        :return: True if any context chunk is in a different language than the answer, False otherwise.
        :rtype: bool
        """
        answer_lang = self._language_detector.detect_language(answer)

        chunk_langs = [
            self._language_detector.detect_language(text)
            for text in context_chunks.values()
        ]

        is_cross_lingual = any(chunk_lang != answer_lang for chunk_lang in chunk_langs)

        return is_cross_lingual

    def compute_bm25_scores(
        self, answer: str, context_chunks: dict[int, str]
    ) -> dict[int, float]:
        """
        Compute BM25 scores for lexical matching

        :param str answer: The generated answer text.
        :param dict[int, str] context_chunks: A dictionary mapping chunk IDs to their text.
        :return: A dictionary mapping chunk IDs to their BM25 scores.
        :rtype: dict[int, float]
        """

        query_tokens = self._tokenizer.encode(answer)
        query_tokens = [str(token) for token in query_tokens]

        tokenized_chunks = [self._tokenizer.encode(c) for c in context_chunks.values()]
        tokenized_chunks = [
            [str(token) for token in chunk] for chunk in tokenized_chunks
        ]

        scores = self._bm25_engine.get_scores(query_tokens, tokenized_chunks)
        return {
            chunk_id: score for chunk_id, score in zip(context_chunks.keys(), scores)
        }

    def compute_embedding_similarities(
        self, answer: str, context_chunks: dict[int, str]
    ) -> dict[int, float]:
        """
        Compute embedding similarities for semantic matching

        :param str answer: The generated answer text.
        :param dict[int, str] context_chunks: A dictionary mapping chunk IDs to their text.
        :return: A dictionary mapping chunk IDs to their similarity scores with the answer.
        :rtype: dict[int, float]
        """
        if not self._embedder:
            raise ValueError("Embedder is not initialized.")

        # Combine answer and chunks into a single batch request
        # Use -1 as the key for the answer to avoid conflicts with chunk IDs
        all_texts = {-1: answer}
        all_texts.update(context_chunks)

        # Get all embeddings in a single API call
        all_embeddings = self._embedder.batch_embed_dict(all_texts)

        # Extract answer embedding
        answer_embedding = all_embeddings[-1]

        # Compute similarities between answer and each chunk
        similarities = {
            chunk_id: self._embedder.similarity(answer_embedding, chunk_embedding)
            for chunk_id, chunk_embedding in all_embeddings.items()
            if chunk_id != -1
        }

        return similarities

    # def normalize_scores(self, scores: dict[int, float]) -> dict[int, float]:
    #     """
    #     Normalize scores to [0, 1] range

    #     :param dict[int, float] scores: A dictionary mapping IDs to their scores.
    #     :return: A dictionary mapping IDs to their normalized scores.
    #     :rtype: dict[int, float]
    #     """
    #     score_values = list(scores.values())
    #     min_score = min(score_values)
    #     max_score = max(score_values)
    #     if max_score - min_score == 0:
    #         return {k: 0.0 for k in scores.keys()}
    #     normalized = {
    #         k: (v - min_score) / (max_score - min_score) for k, v in scores.items()
    #     }
    #     return normalized

    def softmax_normalize(self, scores: dict[int, float]) -> dict[int, float]:
        """Normalize scores using softmax to create a probability distribution"""
        import numpy as np

        score_values = np.array(list(scores.values()))
        exp_scores = np.exp(
            score_values - np.max(score_values)
        )  # Subtract max for numerical stability
        softmax_scores = exp_scores / exp_scores.sum()
        return {k: float(s) for k, s in zip(scores.keys(), softmax_scores)}

    def _combine_scores(
        self,
        bm25_scores: dict[int, float],
        embedding_scores: dict[int, float],
        bm25_weight: float = 0.6,
    ) -> dict[int, float]:
        """Combine BM25 and embedding scores with weighted average"""
        normalized_bm25 = self.softmax_normalize(bm25_scores)
        normalized_embedding = self.softmax_normalize(embedding_scores)

        combined = {
            i: bm25_weight * normalized_bm25[i]
            + (1 - bm25_weight) * normalized_embedding[i]
            for i in normalized_bm25.keys()
        }
        return combined

    def get_citations(
        self, answer: str, context_chunks: dict[int, str]
    ) -> Optional[dict[int, float]]:
        """
        Get citation scores for context chunks based on the answer.

        :param str answer: The generated answer text.
        :param dict[int, str] context_chunks: A dictionary mapping chunk IDs to their text.
        :return: A dictionary mapping chunk IDs to their citation scores or None if no citations are found.
        :rtype: Optional[dict[int, float]]
        """
        if not context_chunks or not answer.strip():
            return None

        # Detect if this is a cross-lingual case
        is_cross_lingual = self._detect_cross_lingual(answer, context_chunks)

        # Choose scoring strategy based on language detection
        if is_cross_lingual:
            # Use semantic matching for cross-lingual cases, fallback to BM25 if embedder is not available
            try:
                scores = self.compute_embedding_similarities(answer, context_chunks)
            # Lower threshold for embeddings
            except ValueError:
                warnings.warn(
                    "Cross-lingual case detected but embedder is not available. The results are likely to be suboptimal. Falling back to BM25 scoring."
                )
                scores = self.compute_bm25_scores(answer, context_chunks)
        elif self.enable_same_language_semantics:
            # Use hybrid matching for same-language cases
            bm25_scores = self.compute_bm25_scores(answer, context_chunks)
            embedding_scores = self.compute_embedding_similarities(
                answer, context_chunks
            )
            scores = self._combine_scores(
                bm25_scores, embedding_scores, bm25_weight=0.6
            )
        else:
            # Use lexical matching for same-language cases (BM25)
            scores = self.compute_bm25_scores(answer, context_chunks)

        normalized_scores = self.softmax_normalize(scores)
        sorted_scores = dict(
            sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
        )

        threshold = elbow_method(list(sorted_scores.values()))

        print("Threshold from elbow method:", threshold)

        final_threshold = max(threshold, self.min_absolute_score)

        selected_citations = {
            chunk_id: score
            for chunk_id, score in sorted_scores.items()
            if score >= final_threshold
        }

        if not selected_citations:
            return None

        return selected_citations
