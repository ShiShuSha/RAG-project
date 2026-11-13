"""Векторное хранилище Chroma и поиск похожих чанков."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import Settings

# Chroma по умолчанию отправляет анонимную телеметрию — в учебном проекте
# она не нужна (и может тормозить запуск без сети).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


@dataclass
class RetrievedChunk:
    """Найденный чанк вместе с оценкой релевантности (0..1)."""

    document: Document
    score: float

    @property
    def text(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict:
        return self.document.metadata


def _chunk_id(chunk: Document, index: int) -> str:
    source = str(chunk.metadata.get("source", ""))
    digest = hashlib.sha1(
        f"{source}|{index}|{chunk.page_content}".encode("utf-8")
    ).hexdigest()
    return digest[:24]


def compute_fingerprint(
    chunks: list[Document],
    settings: Settings,
    embedding_name: str,
) -> str:
    """Отпечаток содержимого базы: позволяет не переиндексировать зря."""
    payload = {
        "version": MANIFEST_VERSION,
        "embedding": embedding_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "collection": settings.collection_name,
        "chunks": [
            {
                "source": str(chunk.metadata.get("source", "")),
                "index": index,
                "chunk_index": chunk.metadata.get("chunk_index"),
                "text": chunk.page_content,
            }
            for index, chunk in enumerate(chunks)
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_manifest(persist_dir: Path) -> dict | None:
    path = persist_dir / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def create_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    settings: Settings,
    *,
    fingerprint: str | None = None,
    rebuild: bool = False,
) -> tuple["Chroma", bool]:
    """Создаёт (или переиспользует) коллекцию Chroma.

    Возвращает пару ``(store, rebuilt)``. Переиндексация выполняется, если
    включён ``rebuild``, изменились документы/модель или база отсутствует.
    """
    from langchain_chroma import Chroma

    persist_dir = settings.persist_dir
    manifest = _read_manifest(persist_dir)
    can_reuse = (
        not rebuild
        and chunks
        and manifest is not None
        and manifest.get("fingerprint") == fingerprint
        and (persist_dir / "chroma.sqlite3").exists()
    )

    if can_reuse:
        logger.info("Использую существующий индекс %s", persist_dir)
        store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        return store, False

    if persist_dir.exists():
        logger.info("Пересоздаю индекс в %s", persist_dir)
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    ids = [_chunk_id(chunk, index) for index, chunk in enumerate(chunks)]
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        collection_name=settings.collection_name,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )

    if fingerprint is not None:
        (persist_dir / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "version": MANIFEST_VERSION,
                    "fingerprint": fingerprint,
                    "collection": settings.collection_name,
                    "chunks": len(chunks),
                    "embedding": settings.embedding_model,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return store, True


def retrieve(
    store: "Chroma",
    query: str,
    k: int = 3,
) -> list[RetrievedChunk]:
    """Возвращает топ-k чанков с релевантностью от 0 (плохо) до 1 (идеально)."""
    query = query.strip()
    if not query:
        return []

    try:
        pairs = store.similarity_search_with_relevance_scores(query, k=k)
    except Exception as exc:  # noqa: BLE001 - совместимость версий Chroma
        logger.debug("relevance-поиск недоступен (%s), использую расстояния", exc)
        pairs = [
            (doc, max(0.0, min(1.0, 1.0 - float(distance))))
            for doc, distance in store.similarity_search_with_score(query, k=k)
        ]

    results = [
        RetrievedChunk(document=document, score=float(max(0.0, min(1.0, score))))
        for document, score in pairs
    ]
    results.sort(key=lambda item: item.score, reverse=True)
    return results
