from enum import Enum
from typing import Any, TypeAlias, Union

import numpy as np

VectorLike: TypeAlias = Union[
    list[float], tuple[float, ...], list[int], tuple[int, ...], np.ndarray[Any, Any]
]


class CitationAlgorithm(str, Enum):
    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
