"""
Embedding Engine wrapper around sentence-transformers.
Provides lazy-loaded embedding model for text-to-vector conversion.
"""

import threading
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.utils.logger import nlp_logger


class EmbeddingEngine:
    """
    Wraps sentence-transformers for text embedding.
    Uses lazy loading to defer model initialization until first use.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model_name
        self._model = None
        # Requests run in worker threads; load the model only once.
        self._load_lock = threading.Lock()
        nlp_logger.info(f"EmbeddingEngine configured with model: {self.model_name}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    nlp_logger.info(f"Loading embedding model: {self.model_name}")
                    self._model = SentenceTransformer(self.model_name)
                    nlp_logger.info("Embedding model loaded successfully")
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string into a vector.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts into vectors.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        nlp_logger.info(f"Batch embedding {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()
