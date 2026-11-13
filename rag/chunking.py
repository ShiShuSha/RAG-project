"""Разбиение документов на чанки с перекрытием."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_splitter(chunk_size: int = 500, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
    """Создаёт сплиттер, учитывающий абзацы и русскую пунктуацию.

    ``RecursiveCharacterTextSplitter`` считает длину в символах, а не в
    токенах. Для русского текста 500 символов — это примерно 70-100
    токенов; параметры можно изменить через переменные окружения.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ": ",
            ", ",
            " ",
            "",
        ],
    )


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Режет документы на чанки и добавляет им стабильные метаданные."""
    if not documents:
        raise ValueError("Нет документов для разбиения")

    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata = {
            **chunk.metadata,
            "chunk_index": index,
            "chunk_chars": len(chunk.page_content),
        }

    if not chunks:
        raise ValueError("После разбиения не осталось ни одного чанка")
    return chunks
