"""Конфигурация проекта.

Все параметры читаются из переменных окружения (или из файла ``.env``),
поэтому один и тот же код работает и с локальной моделью, и с внешним API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Подхватываем .env, если установлен python-dotenv."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # зависимость опциональна
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть целым числом, получено: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть числом, получено: {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "да"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw).expanduser() if raw and raw.strip() else default
    return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass(frozen=True)
class Settings:
    """Все настройки прототипа в одном месте."""

    data_dir: Path
    persist_dir: Path
    collection_name: str

    # Индексация
    chunk_size: int
    chunk_overlap: int

    # Эмбеддинги
    embedding_provider: str
    embedding_model: str
    allow_model_download: bool

    # Поиск
    top_k: int
    min_relevance: float

    # Генерация
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    openai_base_url: str | None
    openai_api_key: str | None
    yandex_folder_id: str | None
    yandex_api_key: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()

        settings = cls(
            data_dir=_env_path("RAG_DATA_DIR", PROJECT_ROOT / "data"),
            persist_dir=_env_path("RAG_CHROMA_DIR", PROJECT_ROOT / ".chroma"),
            collection_name=_env_str("RAG_COLLECTION", "knowledge_base"),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 500),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 50),
            embedding_provider=_env_str("EMBEDDING_PROVIDER", "auto").lower(),
            embedding_model=_env_str("EMBEDDING_MODEL", "cointegrated/rubert-tiny2"),
            allow_model_download=_env_bool("ALLOW_MODEL_DOWNLOAD", True),
            top_k=_env_int("RAG_TOP_K", 3),
            min_relevance=_env_float("RAG_MIN_RELEVANCE", 0.10),
            llm_provider=_env_str("LLM_PROVIDER", "auto").lower(),
            llm_model=_env_str("LLM_MODEL", "llama3.2"),
            ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            yandex_folder_id=os.getenv("YANDEX_FOLDER_ID") or None,
            yandex_api_key=os.getenv("YANDEX_API_KEY") or None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("RAG_CHUNK_SIZE должен быть больше нуля")
        if self.chunk_overlap < 0:
            raise ValueError("RAG_CHUNK_OVERLAP не может быть отрицательным")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP должен быть меньше RAG_CHUNK_SIZE")
        if self.top_k <= 0:
            raise ValueError("RAG_TOP_K должен быть больше нуля")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("RAG_MIN_RELEVANCE должен быть в диапазоне 0..1")
