import numpy as np

from ragcite.core.types import VectorLike


def cosine_similarity(vec1: VectorLike, vec2: VectorLike) -> float:
    """
    Compute the cosine similarity between two vectors.

    :param vec1: First vector.
    :param vec2: Second vector.
    :return: Cosine similarity score.
    :rtype: float
    """
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)
