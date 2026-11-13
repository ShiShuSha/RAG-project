"""Эмбеддинги: sentence-transformers с офлайн-фолбэком.

Основной путь — мультиязычная/русскоязычная модель из Hugging Face.
Если модель не скачана и сети нет, используется детерминированный
hashing-эмбеддер (символьные n-граммы + слова). Это позволяет прототипу
запускаться в полностью изолированном окружении без скачивания весов,
хотя качество поиска у резервного варианта ниже.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from langchain_core.embeddings import Embeddings

from .config import Settings

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]+")


@dataclass
class EmbeddingInfo:
    embeddings: Embeddings
    name: str
    fallback_reason: str | None = None


class HashingEmbeddings(Embeddings):
    """Офлайн-эмбеддер на хешировании признаков (без обучения).

    Признаки: слова, их префиксы (грубая основа для русской морфологии),
    биграммы слов и символьные n-граммы. Векторы L2-нормированы, поэтому
    косинусная близость совпадает со скалярным произведением.
    """

    def __init__(self, dimensions: int = 1024) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions должен быть больше нуля")
        self.dimensions = dimensions

    def _features(self, text: str) -> Counter[str]:
        normalized = text.lower().replace("ё", "е")
        words = _WORD_RE.findall(normalized)
        features: Counter[str] = Counter()

        for word in words:
            features[f"w:{word}"] += 1.0
            # Грубая основа: «сотрудников»/«сотрудники» -> «сотруд»
            if len(word) > 5:
                features[f"p:{word[:6]}"] += 0.7
        for left, right in zip(words, words[1:]):
            features[f"b:{left}_{right}"] += 0.5

        joined = f" {' '.join(words)} "
        for n in (3, 4, 5):
            for start in range(max(0, len(joined) - n + 1)):
                features[f"c:{joined[start:start + n]}"] += 0.25
        return features

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for feature, count in self._features(text).items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * math.log1p(count)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def create_embeddings(settings: Settings) -> EmbeddingInfo:
    """Создаёт эмбеддер согласно настройкам, с понятным фолбэком."""
    provider = settings.embedding_provider
    if provider in {"hashing", "hash", "offline"}:
        return EmbeddingInfo(
            embeddings=HashingEmbeddings(),
            name="hashing-1024 (offline)",
        )
    if provider not in {"auto", "sentence-transformers", "sentence_transformers", "st", "huggingface"}:
        raise ValueError(
            f"Неизвестный EMBEDDING_PROVIDER={provider!r}. "
            "Допустимо: auto, sentence-transformers, hashing."
        )

    errors: list[str] = []
    attempts = [True, False] if settings.allow_model_download else [True]
    for local_only in attempts:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"local_files_only": local_only},
                encode_kwargs={"normalize_embeddings": True},
            )
            embeddings.embed_query("проверка модели эмбеддингов")
            mode = "локальный кэш" if local_only else "загрузка с Hugging Face"
            return EmbeddingInfo(
                embeddings=embeddings,
                name=f"sentence-transformers:{settings.embedding_model} ({mode})",
            )
        except Exception as exc:  # noqa: BLE001 - нужен любой сбой модели
            errors.append(f"local_files_only={local_only}: {type(exc).__name__}: {exc}")

    if provider == "auto":
        reason = errors[-1] if errors else "модель недоступна"
        logger.warning(
            "Не удалось загрузить sentence-transformers модель %r (%s). "
            "Использую офлайн-эмбеддер hashing-1024.",
            settings.embedding_model,
            reason,
        )
        return EmbeddingInfo(
            embeddings=HashingEmbeddings(),
            name="hashing-1024 (offline fallback)",
            fallback_reason=reason,
        )

    raise RuntimeError(
        "Не удалось загрузить модель эмбеддингов "
        f"{settings.embedding_model!r}. Проверьте интернет/кэш Hugging Face "
        "или задайте EMBEDDING_PROVIDER=hashing для офлайн-режима.\n"
        + "\n".join(errors)
    )
