import hashlib

import pytest

from app.core.nlp.clinical_fallback import ClinicalFallbackEngine
from app.core.nlp.vector_store import VectorStore
from app.core.validation.guideline_validator import GuidelineValidator


class FakeEmbeddingEngine:
    """Deterministic unit vectors; avoids loading the real model."""

    model_name = "fake-embedding"

    def embed_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [b - 127.5 for b in digest[:16]]
        norm = sum(v * v for v in vector) ** 0.5
        return [v / norm for v in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


@pytest.fixture
def store(tmp_path):
    vector_store = VectorStore()
    vector_store.persist_dir = tmp_path / "chroma"
    vector_store.collection_name = "test_guidelines"
    vector_store.embedding_engine = FakeEmbeddingEngine()
    return vector_store


@pytest.fixture
def guidelines_dir(tmp_path):
    root = tmp_path / "guidelines"
    (root / "cardiology").mkdir(parents=True)
    (root / "pulmonology").mkdir()
    (root / "cardiology" / "heart.txt").write_text(
        "Heart failure section.\n\nAngina section.", encoding="utf-8"
    )
    (root / "pulmonology" / "lung.txt").write_text(
        "Pneumonia section.\n\nAsthma section.\n\nCOPD section.", encoding="utf-8"
    )
    return root


def test_every_file_is_indexed(store, guidelines_dir):
    """Chunk IDs used to restart at 0 per file, so files overwrote each other."""
    assert store.sync_guidelines(guidelines_dir) == 5
    assert store.get_document_count() == 5

    sources = {m["source"] for m in store.collection.get(include=["metadatas"])["metadatas"]}
    assert sources == {"cardiology/heart.txt", "pulmonology/lung.txt"}


def test_sync_is_idempotent_and_follows_file_changes(store, guidelines_dir):
    store.sync_guidelines(guidelines_dir)
    assert store.sync_guidelines(guidelines_dir) == 0

    (guidelines_dir / "cardiology" / "heart.txt").write_text(
        "Heart failure section, revised.", encoding="utf-8"
    )
    assert store.sync_guidelines(guidelines_dir) == 4
    assert store.get_document_count() == 4


def test_sync_keeps_documents_ingested_through_the_api(store, guidelines_dir):
    store.sync_guidelines(guidelines_dir)
    store.ingest_documents(["Custom note."], [{"source": "api", "origin": "api"}])
    store.ingest_documents(["Custom note."], [{"source": "api", "origin": "api"}])
    assert store.get_document_count() == 6  # re-ingesting the same text is a no-op

    (guidelines_dir / "pulmonology" / "lung.txt").write_text("Pneumonia only.", encoding="utf-8")
    store.sync_guidelines(guidelines_dir)
    assert store.get_document_count() == 4  # 2 cardiology + 1 pulmonology + 1 api


def test_collection_with_wrong_distance_space_is_rebuilt(store, guidelines_dir):
    store.client.create_collection(store.collection_name)  # legacy: default L2
    assert store.uses_expected_distance() is False

    store.sync_guidelines(guidelines_dir)
    assert store.uses_expected_distance() is True
    assert store.get_document_count() == 5


def test_token_validation_score_is_never_negative():
    validator = GuidelineValidator(vector_store=object())
    docs = [
        {"text": "heart failure management", "distance": 1.4},
        {"text": "heart failure diagnosis", "distance": 1.2},
    ]
    consistent, support, _ = validator._validate_with_tokens("Heart Failure", docs)
    assert support == 0.0
    assert consistent is False


def test_retrieval_boost_matches_whole_words_only():
    """'tb' is inside 'heartburn' and 'uti' inside 'routine'."""
    engine = ClinicalFallbackEngine()
    docs = [{"text": "Patients with heartburn should get routine follow-up.", "distance": 0.1}]
    result = engine.diagnose("mild heartburn", docs)

    boosted = set(result["fallback_reasoning"]["retrieval_boost"])
    assert boosted == {"gerd"}
