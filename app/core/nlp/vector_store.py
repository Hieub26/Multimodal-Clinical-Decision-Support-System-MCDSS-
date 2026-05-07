"""
ChromaDB Vector Store for clinical guideline retrieval.
"""

from pathlib import Path
import chromadb
from app.config import settings
from app.core.nlp.embedding_engine import EmbeddingEngine
from app.utils.logger import nlp_logger


class VectorStore:
    """ChromaDB-based vector store for clinical guidelines."""

    def __init__(self):
        self.persist_dir = settings.chroma_persist_dir
        self.collection_name = settings.chroma_collection_name
        self.embedding_engine = EmbeddingEngine()
        self._client = None
        self._collection = None
        nlp_logger.info(f"VectorStore configured: {self.persist_dir}")

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Clinical guidelines for RAG"},
            )
        return self._collection

    def ingest_documents(self, texts: list[str], metadatas: list[dict] = None,
                         chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        all_chunks, all_metadatas, all_ids = [], [], []
        for idx, text in enumerate(texts):
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)
            for ci, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"doc_{idx}_chunk_{ci}")
                meta = metadatas[idx].copy() if metadatas and idx < len(metadatas) else {}
                meta["chunk_index"] = ci
                all_metadatas.append(meta)

        embeddings = self.embedding_engine.embed_batch(all_chunks)
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            end = min(i + batch_size, len(all_chunks))
            self.collection.add(
                ids=all_ids[i:end], documents=all_chunks[i:end],
                embeddings=embeddings[i:end], metadatas=all_metadatas[i:end],
            )
        nlp_logger.info(f"Ingested {len(all_chunks)} chunks")
        return len(all_chunks)

    def ingest_from_file(self, file_path: str) -> int:
        path = Path(file_path)
        if not path.exists():
            nlp_logger.error(f"File not found: {file_path}")
            return 0
        content = path.read_text(encoding="utf-8")
        sections = [s.strip() for s in content.split("\n\n") if s.strip()]
        metadatas = [{"source": path.name, "section": i} for i in range(len(sections))]
        return self.ingest_documents(sections, metadatas)

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.embedding_engine.embed_text(query)
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        search_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                search_results.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })
        return search_results

    def get_document_count(self) -> int:
        return self.collection.count()

    def clear(self):
        self.client.delete_collection(self.collection_name)
        self._collection = None

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                for sep in [". ", ".\n", "\n\n", "\n", " "]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
            chunks.append(text[start:end].strip())
            start = end - overlap
        return [c for c in chunks if c]
