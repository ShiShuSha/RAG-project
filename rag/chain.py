"""RAG-цепочка: поиск -> строгий промпт -> LLM -> ответ с источниками."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser

from .chunking import split_documents
from .config import Settings
from .embeddings import create_embeddings
from .llm import REFUSAL, LLMInfo, build_llm, extractive_answer
from .loaders import discover_documents, load_documents
from .prompts import get_prompt
from .vector_store import (
    RetrievedChunk,
    compute_fingerprint,
    create_vector_store,
    retrieve,
)

logger = logging.getLogger(__name__)


@dataclass
class RagAnswer:
    question: str
    answer: str
    refused: bool
    sources: list[RetrievedChunk] = field(default_factory=list)
    llm_name: str = ""
    best_score: float = 0.0


def format_context(sources: list[RetrievedChunk]) -> str:
    """Превращает чанки в пронумерованный контекст для промпта."""
    blocks: list[str] = []
    for number, source in enumerate(sources, start=1):
        metadata = source.metadata
        file_name = metadata.get("file_name") or Path(
            str(metadata.get("source", "документ"))
        ).name
        chunk_index = metadata.get("chunk_index", "?")
        blocks.append(
            f"[Фрагмент {number} | файл: {file_name} | чанк: {chunk_index} | "
            f"релевантность: {source.score:.3f}]\n{source.text}"
        )
    return "\n\n".join(blocks)


def _looks_like_refusal(answer: str) -> bool:
    normalized = " ".join(answer.lower().replace("ё", "е").split())
    if not normalized:
        return True
    if REFUSAL.lower().replace("ё", "е") in normalized:
        return True
    return "нет информации" in normalized and (
        "документ" in normalized or "контекст" in normalized
    )


class RagPipeline:
    """Готовая RAG-цепочка с возможностью задавать вопросы."""

    def __init__(
        self,
        settings: Settings,
        store,
        llm_info: LLMInfo,
        *,
        embedding_name: str,
        document_count: int,
        chunk_count: int,
        rebuilt: bool,
    ) -> None:
        self.settings = settings
        self.store = store
        self.llm_info = llm_info
        self.embedding_name = embedding_name
        self.document_count = document_count
        self.chunk_count = chunk_count
        self.rebuilt = rebuilt

        self._prompt = get_prompt(REFUSAL)
        self._chain = None
        if llm_info.llm is not None:
            # Именно LangChain Expression Language-цепочка:
            # prompt -> LLM -> строковый парсер.
            self._chain = self._prompt | llm_info.llm | StrOutputParser()

    @classmethod
    def build(cls, settings: Settings, *, rebuild: bool = False) -> "RagPipeline":
        """Загружает документы, индексирует их и собирает цепочку."""
        paths = discover_documents(settings.data_dir)
        logger.info("Найдено документов: %d", len(paths))
        documents = load_documents(paths)
        chunks = split_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        logger.info("Получено чанков: %d", len(chunks))

        embedding_info = create_embeddings(settings)
        fingerprint = compute_fingerprint(chunks, settings, embedding_info.name)
        store, rebuilt = create_vector_store(
            chunks,
            embedding_info.embeddings,
            settings,
            fingerprint=fingerprint,
            rebuild=rebuild,
        )
        logger.info(
            "%s векторный индекс (%s)",
            "Создан" if rebuilt else "Переиспользован",
            embedding_info.name,
        )

        llm_info = build_llm(settings)
        return cls(
            settings=settings,
            store=store,
            llm_info=llm_info,
            embedding_name=embedding_info.name,
            document_count=len(documents),
            chunk_count=len(chunks),
            rebuilt=rebuilt,
        )

    @property
    def llm_name(self) -> str:
        return self.llm_info.name

    def ask(self, question: str) -> RagAnswer:
        """Отвечает на вопрос, возвращая также исходные чанки и их оценки."""
        question = question.strip()
        if not question:
            raise ValueError("Вопрос не может быть пустым")

        sources = retrieve(self.store, question, k=self.settings.top_k)
        best_score = sources[0].score if sources else 0.0

        # Быстрый отказ, если даже лучший чанк почти не похож на вопрос.
        if not sources or best_score < self.settings.min_relevance:
            logger.info(
                "Отказ: лучшая релевантность %.3f < порога %.3f",
                best_score,
                self.settings.min_relevance,
            )
            return RagAnswer(
                question=question,
                answer=REFUSAL,
                refused=True,
                sources=sources,
                llm_name=self.llm_name,
                best_score=best_score,
            )

        context = format_context(sources)

        if self._chain is None:
            extracted = extractive_answer(question, sources)
            refused = extracted is None
            answer = REFUSAL if refused else f"По данным базы знаний: {extracted}"
        else:
            answer = self._invoke_llm(context, question)
            refused = _looks_like_refusal(answer)

        return RagAnswer(
            question=question,
            answer=answer,
            refused=refused,
            sources=sources,
            llm_name=self.llm_name,
            best_score=best_score,
        )

    def _invoke_llm(self, context: str, question: str) -> str:
        try:
            answer = self._chain.invoke({"context": context, "question": question})
        except Exception as exc:  # noqa: BLE001 - показываем понятную ошибку
            raise RuntimeError(
                f"LLM {self.llm_name!r} не смогла сгенерировать ответ: {exc}. "
                "Проверьте, что сервер модели запущен и модель скачана, "
                "либо задайте LLM_PROVIDER=extractive."
            ) from exc
        return str(answer).strip() or REFUSAL

    def ask_many(self, questions: list[str]) -> list[RagAnswer]:
        return [self.ask(question) for question in questions]
