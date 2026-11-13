"""Загрузка документов: TXT/MD, PDF и DOCX."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}


def _read_text(path: Path) -> str:
    """Читаем текстовый файл, не падая на типичных кодировках Windows."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _load_pdf(path: Path) -> tuple[str, dict]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise ImportError(
            "Для чтения PDF установите pypdf: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(page for page in pages if page)
    if not text:
        raise ValueError(
            f"В PDF {path.name!r} не найден текстовый слой. "
            "Скорее всего, документ отсканирован и требует OCR."
        )
    return text, {"pages": len(reader.pages)}


def _load_docx(path: Path) -> tuple[str, dict]:
    try:
        import docx2txt
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise ImportError(
            "Для чтения DOCX установите docx2txt: pip install docx2txt"
        ) from exc

    text = (docx2txt.process(str(path)) or "").strip()
    if not text:
        raise ValueError(f"Не удалось извлечь текст из DOCX {path.name!r}")
    return text, {}


def load_document(path: str | Path) -> Document:
    """Загружает один документ и возвращает LangChain ``Document``."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Документ не найден: {path}")
    if not path.is_file():
        raise ValueError(f"Ожидался файл, получено: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Формат {suffix or '<без расширения>'} не поддерживается. Доступны: {supported}")

    extra: dict = {}
    if suffix in {".txt", ".md"}:
        text = _read_text(path).strip()
    elif suffix == ".pdf":
        text, extra = _load_pdf(path)
    else:
        text, extra = _load_docx(path)

    if not text:
        raise ValueError(f"Документ пуст: {path}")

    metadata = {
        "source": str(path),
        "file_name": path.name,
        "file_type": suffix.lstrip("."),
        **extra,
    }
    return Document(page_content=text, metadata=metadata)


def discover_documents(data_dir: str | Path) -> list[Path]:
    """Рекурсивно находит поддерживаемые документы в каталоге."""
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Каталог с данными не найден: {data_dir}. "
            "Создайте его и положите туда .txt/.pdf/.docx файлы."
        )
    if data_dir.is_file():
        return [data_dir]

    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(data_dir).parts)
    )
    if not paths:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise FileNotFoundError(
            f"В каталоге {data_dir} не найдено документов ({supported})."
        )
    return paths


def load_documents(paths: Iterable[str | Path]) -> list[Document]:
    """Загружает набор документов, пропуская пустые файлы не молча."""
    documents: list[Document] = []
    errors: list[str] = []
    for path in paths:
        try:
            documents.append(load_document(path))
        except Exception as exc:  # noqa: BLE001 - собираем все ошибки сразу
            errors.append(f"{path}: {exc}")
    if errors:
        raise RuntimeError("Не удалось загрузить документы:\n- " + "\n- ".join(errors))
    return documents
