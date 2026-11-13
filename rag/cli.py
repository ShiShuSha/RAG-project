"""CLI демонстрационного RAG-прототипа."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .chain import RagAnswer, RagPipeline
from .config import Settings

DEMO_QUESTIONS = [
    "В каком городе находится главный офис TechNova?",
    "Сколько стоит расширенная подписка на TechNova CRM?",
    "Как связаться с технической поддержкой?",
    # В документе этого нет — модель обязана отказаться отвечать.
    "Кто генеральный директор TechNova?",
]


def _configure_console() -> None:
    """Русский текст в Windows-консоли без кракозябр."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def _preview(text: str, limit: int = 260) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _print_answer(number: int, total: int, result: RagAnswer) -> None:
    print()
    print(f"[{number}/{total}] Вопрос: {result.question}")
    print(f"Ответ: {result.answer}")
    verdict = "ОТКАЗ (нет данных в базе)" if result.refused else "ответ найден"
    print(f"Статус: {verdict} | лучшая релевантность: {result.best_score:.3f}")
    if not result.sources:
        print("Источники: ничего не найдено")
        return
    print("Источники (чанки, на которых основан ответ):")
    for index, source in enumerate(result.sources, start=1):
        metadata = source.metadata
        file_name = metadata.get("file_name") or Path(
            str(metadata.get("source", "документ"))
        ).name
        chunk_index = metadata.get("chunk_index", "?")
        print(
            f"  {index}. {file_name}, чанк #{chunk_index} — "
            f"релевантность {source.score:.3f}"
        )
        print(f"     «{_preview(source.text)}»")


def _answer_to_dict(result: RagAnswer) -> dict:
    return {
        "question": result.question,
        "answer": result.answer,
        "refused": result.refused,
        "best_score": round(result.best_score, 4),
        "llm": result.llm_name,
        "sources": [
            {
                "file": source.metadata.get("file_name")
                or source.metadata.get("source"),
                "chunk_index": source.metadata.get("chunk_index"),
                "relevance": round(source.score, 4),
                "text": source.text,
            }
            for source in result.sources
        ],
    }


def _print_header(pipeline: RagPipeline) -> None:
    print("=" * 72)
    print("RAG-прототип: база знаний компании TechNova")
    print("=" * 72)
    print(f"Документов: {pipeline.document_count} | чанков: {pipeline.chunk_count}")
    print(f"Эмбеддинги: {pipeline.embedding_name}")
    print(f"LLM: {pipeline.llm_name}")
    print(f"Индекс: {'создан заново' if pipeline.rebuilt else 'переиспользован'}")
    if pipeline.llm_info.fallback_reason:
        print("LLM недоступна — ответы собираются extractive-методом из чанков.")


def _interactive_loop(pipeline: RagPipeline) -> None:
    print()
    print("Интерактивный режим. Введите вопрос или 'выход' для завершения.")
    while True:
        try:
            question = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"выход", "exit", "quit", "q"}:
            return
        result = pipeline.ask(question)
        print(f"Ответ: {result.answer}")
        print(f"Релевантность: {result.best_score:.3f} | отказ: {'да' if result.refused else 'нет'}")
        for index, source in enumerate(result.sources, start=1):
            file_name = source.metadata.get("file_name") or source.metadata.get("source")
            print(
                f"  [{index}] {file_name}, чанк #{source.metadata.get('chunk_index', '?')}"
                f" — {source.score:.3f}: «{_preview(source.text, 160)}»"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG-прототип на LangChain + Chroma + sentence-transformers"
    )
    parser.add_argument(
        "--ask",
        action="append",
        metavar="ВОПРОС",
        help="задать вопрос (можно указать несколько раз)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="интерактивный режим вопросов-ответов",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="пересоздать векторный индекс, даже если документы не менялись",
    )
    parser.add_argument("--data", metavar="КАТАЛОГ", help="каталог с документами")
    parser.add_argument("--top-k", type=int, metavar="N", help="сколько чанков извлекать (по умолчанию 3)")
    parser.add_argument("--json", action="store_true", help="вывести результаты в JSON")
    parser.add_argument(
        "--verbose", action="store_true", help="подробные логи индексации и поиска"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    # Шумные библиотеки не должны забивать вывод.
    for noisy in ("chromadb", "httpx", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = Settings.from_env()
    if args.data:
        settings = replace(settings, data_dir=Path(args.data).expanduser().resolve())
    if args.top_k is not None:
        settings = replace(settings, top_k=args.top_k)
    settings.validate()

    try:
        pipeline = RagPipeline.build(settings, rebuild=args.rebuild)
    except Exception as exc:  # noqa: BLE001 - CLI должен объяснить проблему
        print(f"Ошибка подготовки базы знаний: {exc}", file=sys.stderr)
        return 1

    questions = list(args.ask or [])
    if not questions and not args.interactive:
        questions = list(DEMO_QUESTIONS)

    results: list[RagAnswer] = []
    if questions:
        try:
            results = pipeline.ask_many(questions)
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка генерации ответа: {exc}", file=sys.stderr)
            return 1

    if args.json:
        print(
            json.dumps(
                {
                    "documents": pipeline.document_count,
                    "chunks": pipeline.chunk_count,
                    "embedding_model": pipeline.embedding_name,
                    "llm": pipeline.llm_name,
                    "answers": [_answer_to_dict(result) for result in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_header(pipeline)
        for number, result in enumerate(results, start=1):
            _print_answer(number, len(results), result)

    if args.interactive:
        _interactive_loop(pipeline)
    return 0
