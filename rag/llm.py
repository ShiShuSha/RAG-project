"""Выбор LLM и офлайн-генератор ответов.

Реальная модель (Ollama / OpenAI-совместимый API / YandexGPT) используется,
если она доступна. Иначе прототип не падает, а переходит в extractive-режим:
ответ собирается из предложений найденных чанков без «фантазий».
"""

from __future__ import annotations

import logging
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from .config import Settings
from .vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

REFUSAL = "В предоставленных документах нет информации для ответа на этот вопрос."

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]+")

# Вопросительные слова и служебная лексика не помогают выбрать ответ.
_STOPWORDS = {
    "а", "без", "более", "бы", "был", "была", "были", "было", "быть",
    "в", "вам", "вас", "весь", "во", "вот", "все", "всего", "вы", "где",
    "да", "даже", "для", "до", "его", "ее", "ей", "ему", "если", "есть",
    "еще", "же", "за", "здесь", "и", "из", "или", "им", "их", "к", "как",
    "какая", "какие", "каким", "каких", "какой", "каком", "какое", "какую",
    "когда", "кто", "ли",
    "мне", "может", "можно", "на", "над", "надо", "наш", "не", "него",
    "нее", "нет", "ни", "них", "но", "ну", "о", "об", "однако", "он",
    "она", "они", "оно", "от", "по", "под", "при", "про", "с", "сам",
    "свою", "себе", "так", "также", "такой", "там", "те", "тем", "то",
    "того", "тоже", "той", "только", "том", "тот", "ту", "у", "уже",
    "чего", "чей", "чем", "что", "чтобы", "чье", "эта", "эти", "это",
    "этого", "этой", "этом", "этот", "эту", "я", "сколько", "какой-то",
}


def _stem(token: str) -> str:
    """Очень грубая основа слова — достаточная для сопоставления в демо."""
    token = token.lower().replace("ё", "е")
    return token[:5] if len(token) > 5 else token


def _content_stems(text: str) -> set[str]:
    return {
        _stem(token)
        for token in _TOKEN_RE.findall(text)
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    }


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]


def extractive_answer(question: str, chunks: list[RetrievedChunk]) -> str | None:
    """Собирает ответ из предложений контекста; None — если ответа нет.

    Предложение считается ответом, только если оно покрывает не менее 60%
    значимых слов вопроса. Это защищает от ответа на вопрос, которого в
    базе знаний нет (например, «кто генеральный директор?»).
    """
    query_stems = _content_stems(question)
    if not query_stems:
        return None

    candidates: list[tuple[float, float, int, str]] = []
    for chunk_rank, chunk in enumerate(chunks):
        for sentence_rank, sentence in enumerate(_split_sentences(chunk.text)):
            sentence_stems = _content_stems(sentence)
            if not sentence_stems:
                continue
            coverage = len(query_stems & sentence_stems) / len(query_stems)
            if coverage < 0.6:
                continue
            # При равном покрытии предпочитаем более релевантный чанк.
            score = coverage + 0.1 * chunk.score - 0.01 * chunk_rank - 0.001 * sentence_rank
            candidates.append((score, coverage, chunk_rank, sentence))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    best_score, best_coverage, best_chunk, best_sentence = candidates[0]

    # Добавляем соседнее предложение из того же чанка, если оно тоже
    # хорошо покрывает вопрос (например, цена + что входит в подписку).
    selected = [best_sentence]
    for score, coverage, chunk_rank, sentence in candidates[1:]:
        if chunk_rank != best_chunk or sentence in selected:
            continue
        if coverage >= max(0.6, best_coverage - 0.2):
            selected.append(sentence)
        break

    return " ".join(selected)


def _ollama_models(base_url: str) -> list[str]:
    """Возвращает список локальных моделей Ollama (пусто, если сервер недоступен)."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=1.5) as response:
            if response.status != 200:
                return []
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return []
    return [
        str(item.get("name") or item.get("model"))
        for item in payload.get("models", [])
        if item.get("name") or item.get("model")
    ]


def _resolve_ollama_model(models: list[str], preferred: str) -> str:
    """Берёт настроенную модель, иначе первую доступную локальную."""
    if preferred in models:
        return preferred
    if f"{preferred}:latest" in models:
        return f"{preferred}:latest"
    chosen = models[0]
    logger.warning(
        "Модель Ollama %r не найдена; использую доступную %r. "
        "Задайте OLLAMA_MODEL, чтобы выбрать другую.",
        preferred,
        chosen,
    )
    return chosen



@dataclass
class LLMInfo:
    llm: BaseChatModel | None
    name: str
    fallback_reason: str | None = None


def build_llm(settings: Settings) -> LLMInfo:
    """Создаёт LLM выбранного провайдера или включает extractive-режим."""
    provider = settings.llm_provider
    if provider in {"extractive", "offline", "none", "noop"}:
        return LLMInfo(llm=None, name="extractive (offline)")
    if provider not in {"auto", "ollama", "openai", "yandex"}:
        raise ValueError(
            f"Неизвестный LLM_PROVIDER={provider!r}. "
            "Допустимо: auto, ollama, openai, yandex, extractive."
        )

    errors: list[str] = []

    def try_ollama() -> LLMInfo | None:
        models = _ollama_models(settings.ollama_base_url)
        if not models:
            errors.append(
                f"Ollama недоступна по адресу {settings.ollama_base_url} "
                "(или не установлена ни одна модель)"
            )
            return None
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            errors.append("пакет langchain-ollama не установлен")
            return None
        preferred = os.getenv("OLLAMA_MODEL", settings.llm_model)
        model = _resolve_ollama_model(models, preferred)
        return LLMInfo(
            llm=ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=0),
            name=f"ollama:{model}",
        )

    def try_openai() -> LLMInfo | None:
        if not settings.openai_api_key:
            errors.append("OPENAI_API_KEY не задан")
            return None
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            errors.append("пакет langchain-openai не установлен")
            return None
        model = os.getenv("OPENAI_MODEL", settings.llm_model)
        try:
            return LLMInfo(
                llm=ChatOpenAI(
                    model=model,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    temperature=0,
                ),
                name=f"openai-compatible:{model}",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"OpenAI-compatible API: {exc}")
            return None

    def try_yandex() -> LLMInfo | None:
        if not (settings.yandex_api_key and settings.yandex_folder_id):
            errors.append("YANDEX_API_KEY/YANDEX_FOLDER_ID не заданы")
            return None
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            errors.append("пакет langchain-openai не установлен")
            return None
        yandex_model = os.getenv("YANDEX_MODEL", "yandexgpt-lite/latest")
        try:
            # YandexGPT предоставляет OpenAI-совместимый endpoint.
            return LLMInfo(
                llm=ChatOpenAI(
                    model=f"gpt://{settings.yandex_folder_id}/{yandex_model}",
                    api_key=settings.yandex_api_key,
                    base_url="https://llm.api.cloud.yandex.net/v1",
                    temperature=0,
                ),
                name=f"yandexgpt:{yandex_model}",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"YandexGPT: {exc}")
            return None

    if provider == "ollama":
        info = try_ollama()
        if info:
            return info
        raise RuntimeError(
            "Ollama недоступна. Запустите `ollama serve` и `ollama pull llama3.2` "
            "или используйте LLM_PROVIDER=extractive."
        )
    if provider == "openai":
        info = try_openai()
        if info:
            return info
        raise RuntimeError(
            "OpenAI-совместимый провайдер не настроен: " + "; ".join(errors)
        )
    if provider == "yandex":
        info = try_yandex()
        if info:
            return info
        raise RuntimeError("YandexGPT не настроен: " + "; ".join(errors))

    # auto: сначала локальная Ollama, затем облачные API, затем офлайн-режим.
    for factory in (try_ollama, try_openai, try_yandex):
        info = factory()
        if info:
            logger.info("Использую LLM: %s", info.name)
            return info

    reason = "; ".join(errors)
    logger.info(
        "Внешняя LLM не найдена (%s). Включаю extractive-режим: ответ будет "
        "собран из найденных чанков без генерации.",
        reason,
    )
    return LLMInfo(
        llm=None,
        name="extractive (offline fallback)",
        fallback_reason=reason,
    )
