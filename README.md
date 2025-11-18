# 📚 Ragcite

A powerful citation toolkit for Retrieval-Augmented Generation (RAG) systems that automatically identifies and scores relevant source chunks used to generate answers.

## ✨ Features

- 🎯 **Multiple Scoring Algorithms**: Supports BM25 (lexical), semantic similarity, and hybrid approaches
- 🌍 **Cross-Lingual Support**: Automatically detects and handles multilingual contexts
- 🔌 **Extensible Architecture**: Easy to integrate custom embedders, tokenizers, and language detectors
- 📊 **Smart Citation Selection**: Uses elbow method to intelligently determine citation thresholds
- ⚡ **Efficient Batch Processing**: Optimized API calls for embedding generation
- 🛠️ **Production Ready**: Built with type safety and comprehensive interfaces

## 🚀 Installation

```bash
pip install ragcite
```

Or using `uv`:

```bash
uv pip install ragcite
```

## 📖 Quick Start

### Basic Usage

```python
from ragcite import Ragcite

# Initialize with default settings (uses OpenAI embeddings)
ragcite = Ragcite()

# Your generated answer
answer = "Paris is the capital of France and is known for the Eiffel Tower."

# Context chunks that were used for generation
context_chunks = {
    0: "Paris is the capital city of France.",
    1: "The Eiffel Tower is located in Paris.",
    2: "London is the capital of the United Kingdom.",
}

# Get citations with scores
citations = ragcite.get_citations(answer, context_chunks)
print(citations)
# Output: {0: 0.95, 1: 0.87}
```

### Without Embeddings (BM25 Only)

```python
from ragcite import Ragcite

# Disable embedding for faster, lexical-only matching
ragcite = Ragcite(enable_embedding=False)

citations = ragcite.get_citations(answer, context_chunks)
```

### Cross-Lingual Citation

```python
from ragcite import Ragcite

ragcite = Ragcite()

# English answer with mixed language context
answer = "The capital of France is Paris."
context_chunks = {
    0: "La capitale de la France est Paris.",  # French
    1: "Paris is a beautiful city.",           # English
    2: "Tokyo is the capital of Japan.",       # English
}

# Automatically detects cross-lingual scenario and uses semantic matching
citations = ragcite.get_citations(answer, context_chunks)
```

### Same-Language Semantic Matching

```python
from ragcite import Ragcite

# Enable hybrid matching for same-language contexts
ragcite = Ragcite(enable_same_language_semantics=True)

# Uses both BM25 and semantic similarity for better results
citations = ragcite.get_citations(answer, context_chunks)
```

## 🔧 Advanced Configuration

### Custom Embedder

```python
from ragcite import Ragcite
from ragcite.core.interfaces import BaseEmbedder
from ragcite.integrations.openai.embedder import OpenAIEmbedder

# Use OpenAI with specific configuration
embedder = OpenAIEmbedder(
    model="text-embedding-3-small",
    api_key="your-api-key"
)

ragcite = Ragcite(embedder=embedder)
```

### Custom Components

```python
from ragcite import Ragcite
from ragcite.integrations.langdetect import LangDetectLanguageDetector
from ragcite.integrations.rank_bm25 import BM25OkEngine
from ragcite.integrations.tiktoken import TiktokenTokenizer

ragcite = Ragcite(
    language_detector=LangDetectLanguageDetector(),
    bm25_engine=BM25OkEngine(),
    tokenizer=TiktokenTokenizer(encoding="cl100k_base")
)
```

## 📊 How It Works

1. **Language Detection**: Automatically detects if the answer and context chunks are in different languages
2. **Score Calculation**: 
   - Cross-lingual: Uses semantic embeddings
   - Same-language (default): Uses BM25 lexical matching
   - Same-language (semantic enabled): Uses hybrid approach combining BM25 and embeddings
3. **Score Normalization**: Normalizes all scores to [0, 1] range
4. **Smart Thresholding**: Applies elbow method to determine optimal citation threshold
5. **Citation Selection**: Returns only the most relevant chunks above the threshold

## 🏗️ Architecture

Ragcite uses a plugin-based architecture with well-defined interfaces:

- **BaseEmbedder**: For generating text embeddings
- **BaseLanguageDetector**: For detecting text language
- **BaseBM25Engine**: For BM25 scoring
- **BaseTokenizer**: For text tokenization

You can implement these interfaces to integrate your own components.

## 🔌 Integrations

### Built-in Integrations

- **OpenAI**: Text embeddings (default)
- **LangDetect**: Language detection (default)
- **Rank-BM25**: BM25 scoring (default)
- **Tiktoken**: OpenAI tokenization (default)

### Creating Custom Integrations

```python
from ragcite.core.interfaces import BaseEmbedder
from ragcite.core.types import VectorLike

class CustomEmbedder(BaseEmbedder):
    def embed(self, text: str) -> VectorLike:
        # Your embedding logic
        pass
    
    def batch_embed(self, texts: list[str]) -> list[VectorLike]:
        # Batch embedding logic
        pass
    
    def similarity(self, vec1: VectorLike, vec2: VectorLike) -> float:
        # Similarity calculation
        pass
```

## 📝 API Reference

### Ragcite

Main class for citation generation.

#### Constructor

```python
Ragcite(
    embedder: Optional[BaseEmbedder] = None,
    language_detector: Optional[BaseLanguageDetector] = None,
    bm25_engine: Optional[BaseBM25Engine] = None,
    tokenizer: Optional[BaseTokenizer] = None,
    enable_embedding: bool = True,
    enable_same_language_semantics: bool = False
)
```

#### Methods

**`get_citations(answer: str, context_chunks: dict[int, str]) -> Optional[dict[int, float]]`**

Returns citation scores for relevant context chunks.

**`compute_bm25_scores(answer: str, context_chunks: dict[int, str]) -> dict[int, float]`**

Computes BM25 lexical matching scores.

**`compute_embedding_similarities(answer: str, context_chunks: dict[int, str]) -> dict[int, float]`**

Computes semantic similarity scores using embeddings.

**`normalize_scores(scores: dict[int, float]) -> dict[int, float]`**

Normalizes scores to [0, 1] range.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [OpenAI](https://openai.com) embeddings
- Uses [Rank-BM25](https://github.com/dorianbrown/rank_bm25) for lexical matching
- Language detection powered by [LangDetect](https://github.com/Mimino666/langdetect)
- Tokenization via [Tiktoken](https://github.com/openai/tiktoken)

## 📮 Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/yourusername/ragcite).

---

Made with ❤️ for the RAG community
