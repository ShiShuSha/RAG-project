"""Небольшой RAG-прототип на LangChain + Chroma.

Пакет намеренно не импортирует тяжёлые зависимости (torch, chromadb) на
уровне модуля: благодаря этому ``import rag`` остаётся быстрым, а ошибки
установки библиотек появляются только там, где они действительно нужны.
"""

from .chain import REFUSAL, RagAnswer, RagPipeline, RetrievedChunk
from .config import Settings

__all__ = [
    "REFUSAL",
    "RagAnswer",
    "RagPipeline",
    "RetrievedChunk",
    "Settings",
]
