"""
ChromaDB Vector Store for clinical guideline retrieval.
"""

import hashlib
from pathlib import Path
import chromadb
from app.config import settings
from app.core.nlp.embedding_engine import EmbeddingEngine
from app.utils.logger import nlp_logger

# Embeddings are L2-normalised, so with the cosine space
# distance = 1 - cosine similarity. Every caller that turns a distance into a
# similarity score (fallback retrieval boost, token validator) relies on this.
DISTANCE_SPACE = "cosine"

# Metadata tag for chunks that mirror files in data/guidelines. Lets a re-sync
# replace them without touching documents ingested through the API.
GUIDELINE_FILE_ORIGIN = "guideline_file"

_FINGERPRINT_FILE = "guidelines.fingerprint"


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
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Clinical guidelines for RAG",
                    "hnsw:space": DISTANCE_SPACE,
                },
            )
        return self._collection

    def uses_expected_distance(self) -> bool:
        """False for collections created before the cosine space was enforced."""
        return (self.collection.metadata or {}).get("hnsw:space") == DISTANCE_SPACE

    def ingest_documents(self, texts: list[str], metadatas: list[dict] = None,
                         chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        # Keyed by ID so identical chunks inside one call collapse instead of
        # raising a duplicate-ID error.
        records: dict[str, tuple[str, dict]] = {}
        for idx, text in enumerate(texts):
            base_meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            source = str(base_meta.get("source", "inline"))
            for ci, chunk in enumerate(self._chunk_text(text, chunk_size, chunk_overlap)):
                meta = base_meta.copy()
                meta["chunk_index"] = ci
                records[self._chunk_id(source, chunk)] = (chunk, meta)

        if not records:
            return 0

        all_ids = list(records)
        all_chunks = [records[i][0] for i in all_ids]
        all_metadatas = [records[i][1] for i in all_ids]

        embeddings = self.embedding_engine.embed_batch(all_chunks)
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            end = min(i + batch_size, len(all_chunks))
            self.collection.upsert(
                ids=all_ids[i:end], documents=all_chunks[i:end],
                embeddings=embeddings[i:end], metadatas=all_metadatas[i:end],
            )
        nlp_logger.info(f"Ingested {len(all_chunks)} chunks")
        return len(all_chunks)

    def ingest_from_file(self, file_path: str, source: str = None,
                         origin: str = None) -> int:
        path = Path(file_path)
        if not path.exists():
            nlp_logger.error(f"File not found: {file_path}")
            return 0
        content = path.read_text(encoding="utf-8")
        sections = [s.strip() for s in content.split("\n\n") if s.strip()]
        metadatas = []
        for i in range(len(sections)):
            meta = {"source": source or path.name, "section": i}
            if origin:
                meta["origin"] = origin
            metadatas.append(meta)
        return self.ingest_documents(sections, metadatas)

    def sync_guidelines(self, guidelines_dir: Path) -> int:
        """Bring the index in line with the guideline files on disk.

        Re-ingests when the files (or the embedding model) changed since the
        last sync, and rebuilds the collection when it was created with a
        different distance space or embedding model.

        Returns:
            Number of chunks ingested (0 when the index was already current).
        """
        guidelines_dir = Path(guidelines_dir)
        files = sorted(guidelines_dir.rglob("*.txt"))
        model_name = self.embedding_engine.model_name
        fingerprint = self._guideline_fingerprint(files, guidelines_dir)
        stored_model, stored_fingerprint = self._read_fingerprint()

        if not self.uses_expected_distance():
            nlp_logger.warning(
                "Guideline collection was not created with the "
                f"'{DISTANCE_SPACE}' distance space — rebuilding it"
            )
            self.clear()
        elif stored_model is not None and stored_model != model_name:
            nlp_logger.warning(
                f"Embedding model changed ({stored_model} -> {model_name}) "
                "— rebuilding guideline collection"
            )
            self.clear()
        elif self.get_document_count() > 0 and stored_fingerprint == fingerprint:
            return 0
        else:
            self.collection.delete(where={"origin": GUIDELINE_FILE_ORIGIN})

        total = 0
        for file_path in files:
            count = self.ingest_from_file(
                str(file_path),
                source=file_path.relative_to(guidelines_dir).as_posix(),
                origin=GUIDELINE_FILE_ORIGIN,
            )
            nlp_logger.info(f"Ingested {count} chunks from {file_path.name}")
            total += count

        self._fingerprint_path.write_text(
            f"{model_name}\n{fingerprint}", encoding="utf-8"
        )
        return total

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
        self._fingerprint_path.unlink(missing_ok=True)

    @property
    def _fingerprint_path(self) -> Path:
        return Path(self.persist_dir) / _FINGERPRINT_FILE

    def _read_fingerprint(self) -> tuple[str | None, str | None]:
        """Return (embedding model, guideline hash) recorded by the last sync."""
        try:
            model_name, fingerprint = (
                self._fingerprint_path.read_text(encoding="utf-8").split("\n", 1)
            )
            return model_name, fingerprint
        except (OSError, ValueError):
            return None, None

    @staticmethod
    def _guideline_fingerprint(files: list[Path], root: Path) -> str:
        digest = hashlib.sha256()
        for file_path in files:
            digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(file_path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _chunk_id(source: str, chunk: str) -> str:
        """Stable ID from source + content: distinct sources never collide and
        re-ingesting the same text overwrites itself instead of duplicating."""
        return hashlib.sha256(f"{source}\n{chunk}".encode("utf-8")).hexdigest()[:32]

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
