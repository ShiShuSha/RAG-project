# RAG-прототип: база знаний компании TechNova

Учебный проект: ответы на вопросы по небольшой базе знаний с помощью
Retrieval-Augmented Generation. Стек: **LangChain + Chroma + sentence-transformers**
и любая LLM (локальная Ollama, OpenAI-совместимый API или YandexGPT).

## Что реализовано

| Этап | Реализация |
|---|---|
| 1.1 Загрузка TXT/MD, PDF, DOCX | `rag/loaders.py` (`pypdf`, `docx2txt`) |
| 1.2 Чанкинг с перекрытием | `rag/chunking.py`, `RecursiveCharacterTextSplitter`, 500/50 |
| 1.3 Векторизация | `rag/embeddings.py`, `cointegrated/rubert-tiny2` (sentence-transformers) |
| 1.4 Векторная БД | `rag/vector_store.py`, Chroma + persist в `.chroma/` |
| 2 Retriever (топ-3) | `retrieve()` + косинусная релевантность 0..1 |
| 2 Prompt Template | `rag/prompts.py`: системная инструкция + чанки + вопрос |
| 2 LLM | `rag/llm.py`: Ollama / OpenAI / YandexGPT / extractive |
| 2 Цепочка | `rag/chain.py`: кастомная LCEL-цепочка `prompt | llm | parser` |
| 5–6 Вопросы и отказ | `python main.py`: 3 вопроса + вопрос без ответа |
| ★ Релевантность и чанки | каждый ответ содержит топ-3 чанка и их оценки |

## Быстрый старт

Нужен Python 3.10+ (проверено на 3.12).

```powershell
# из корня проекта
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

При первом запуске модель эмбеддингов `cointegrated/rubert-tiny2` (~120 МБ)
скачается с Hugging Face и закэшируется.

### Если venv уже сломан

Проще пересоздать окружение, чем чинить несовместимые версии:

```powershell
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\python.exe -m pip install -U pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py
```

## Запуск

```powershell
python main.py                                  # демо: 3 ответа + 1 отказ
python main.py --ask "Сколько сотрудников в компании?"
python main.py --ask "..." --ask "..." --json    # JSON с источниками
python main.py --interactive                     # свободные вопросы
python main.py --rebuild                         # пересоздать индекс
python main.py --verbose                         # логи индексации
```

Пример вывода:

```text
[1/4] Вопрос: В каком городе находится главный офис TechNova?
Ответ: Город Казань.
Статус: ответ найден | лучшая релевантность: 0.707
Источники (чанки, на которых основан ответ):
  1. company.txt, чанк #1 — релевантность 0.707
     «Компания TechNova была основана в 2020 году. Главный офис компании...»

[4/4] Вопрос: Кто генеральный директор TechNova?
Ответ: В предоставленных документах нет информации для ответа на этот вопрос.
Статус: ОТКАЗ (нет данных в базе) | лучшая релевантность: 0.610
```

Отказ работает на двух уровнях:

1. **Порог релевантности** - если лучший чанк слишком далёк от вопроса,
   LLM вообще не вызывается (`RAG_MIN_RELEVANCE`).
2. **Строгий промпт** - модели запрещено использовать знания вне контекста;
   при отсутствии ответа она обязана вернуть фиксированную фразу отказа.
   В extractive-режиме отказ определяет проверка покрытия слов вопроса.

## Выбор LLM

По умолчанию `LLM_PROVIDER=auto`: сначала ищется локальная Ollama, затем
OpenAI-совместимый API, затем YandexGPT. Если ничего не доступно, включается
**extractive-режим**: ответ собирается из предложений найденных чанков без
генерации. Прототип остаётся работоспособным даже без модели.

### Ollama (локально, бесплатно)

```powershell
ollama serve
ollama pull qwen2.5:3b
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="qwen2.5:3b"
python main.py
```

Если указанной модели нет, но в Ollama есть другие, прототип возьмёт первую
доступную и предупредит об этом.

### OpenAI или любой совместимый API

```powershell
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_BASE_URL="https://api.openai.com/v1"   # можно не задавать
$env:OPENAI_MODEL="gpt-4o-mini"
python main.py
```

### YandexGPT

```powershell
$env:LLM_PROVIDER="yandex"
$env:YANDEX_API_KEY="..."
$env:YANDEX_FOLDER_ID="..."
$env:YANDEX_MODEL="yandexgpt-lite/latest"
python main.py
```

YandexGPT используется через OpenAI-совместимый endpoint
`https://llm.api.cloud.yandex.net/v1`.

Смена LLM не требует изменений в цепочке: `rag/chain.py` работает с любым
`BaseChatModel` из LangChain.

## Своя база знаний

Положите файлы `.txt`, `.md`, `.pdf` или `.docx` в `data/` (можно в подпапки)
и запустите `python main.py --rebuild`. Документы индексируются рекурсивно.
PDF должен иметь текстовый слой (сканы без OCR не поддерживаются).

Текущая база — `data/company.txt`: 4 абзаца о вымышленной компании TechNova
(офис, сотрудники, продукт и цены, график и поддержка). В тексте специально
есть факт-«ловушка»: компания не разрабатывает компьютерные игры.

## Настройки

Все параметры читаются из переменных окружения или `.env`
(шаблон — `.env.example`). Основные:

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `RAG_DATA_DIR` | `data` | каталог с документами |
| `RAG_CHROMA_DIR` | `.chroma` | каталог Chroma |
| `RAG_CHUNK_SIZE` | `500` | размер чанка (символы) |
| `RAG_CHUNK_OVERLAP` | `50` | перекрытие (символы) |
| `RAG_TOP_K` | `3` | сколько чанков извлекать |
| `RAG_MIN_RELEVANCE` | `0.10` | порог отказа |
| `EMBEDDING_PROVIDER` | `auto` | `auto` / `sentence-transformers` / `hashing` |
| `EMBEDDING_MODEL` | `cointegrated/rubert-tiny2` | модель эмбеддингов |
| `ALLOW_MODEL_DOWNLOAD` | `true` | разрешить скачивание весов |
| `LLM_PROVIDER` | `auto` | `auto` / `ollama` / `openai` / `yandex` / `extractive` |

`RecursiveCharacterTextSplitter` измеряет длину в **символах**, а не токенах.
500 символов ≈ 70–100 токенов для русского текста. Если нужны именно
~500 токенов, поставьте `RAG_CHUNK_SIZE=1500` (и, например,
`RAG_CHUNK_OVERLAP=150`). Для этой маленькой базы 500 символов удобнее:
каждый абзац становится отдельным чанком.

## Полностью офлайн-режим

```powershell
$env:EMBEDDING_PROVIDER="hashing"
$env:LLM_PROVIDER="extractive"
python main.py
```

- `hashing` - детерминированные векторы на словах, префиксах, биграммах и
  символьных n-граммах, без скачивания моделей;
- `extractive` - ответ из предложений найденных чанков, без генерации.

Качество ниже, чем у sentence-transformers + LLM, но прототип запускается
в изолированном окружении без сети. При недоступности модели в режиме
`auto` фолбэк происходит автоматически.

## Структура проекта

```text
main.py                 точка входа
rag/
  config.py             настройки из env/.env
  loaders.py            TXT/MD/PDF/DOCX
  chunking.py           RecursiveCharacterTextSplitter
  embeddings.py         sentence-transformers + hashing-фолбэк
  vector_store.py       Chroma, манифест индекса, поиск с оценками
  prompts.py            строгий системный промпт
  llm.py                провайдеры LLM + extractive-ответ
  chain.py              RAG-цепочка и результат с источниками
  cli.py                демонстрация и интерактивный режим
data/company.txt        база знаний
tests/test_smoke.py     тесты без сети
```

Индекс переиспользуется между запусками. Отпечаток содержимого документов,
параметров чанкинга и модели эмбеддингов хранится в
`.chroma/manifest.json`; при изменении данных индекс пересоздаётся
автоматически. `--rebuild` форсирует переиндексацию.

## Тесты

```powershell
python -m unittest discover -s tests -v
```

Тесты работают без сети: принудительно включены `hashing`-эмбеддинги и
extractive-режим. Проверяются загрузка/чанкинг, детерминированность векторов,
ответы по тексту и отказ на вопрос, которого в базе нет.

## Если снова ошибка импорта

Проверьте, что запускаете именно Python из venv:

```powershell
.\venv\Scripts\python.exe -c "import langchain, chromadb, sentence_transformers; print('ok')"
```

Типичные причины:

- `ModuleNotFoundError: langchain_community` - в старом коде; здесь эта
  зависимость не используется.
- `ImportError: cannot import name 'RetrievalQA'` - в LangChain 1.x класс
  удалён из `langchain.chains`; здесь вместо него кастомная LCEL-цепочка.
- `No module named 'rag'` - запускайте `python main.py` из корня проекта,
  а не из другой папки.
- Ошибка загрузки модели эмбеддингов без интернета - включите
  `EMBEDDING_PROVIDER=hashing` или заранее скачайте модель.
