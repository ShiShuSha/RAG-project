"""Точка входа RAG-прототипа.

Запуск:
    python main.py                 # демонстрация: 3 вопроса + вопрос без ответа
    python main.py --interactive    # свободные вопросы
    python main.py --ask "..."      # один вопрос
"""

from __future__ import annotations

import os

# Отключаем телеметрию и лишний параллелизм до импорта тяжёлых библиотек.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from rag.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
