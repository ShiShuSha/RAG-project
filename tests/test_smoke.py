"""Быстрые smoke-тесты без сети и без скачивания моделей.

Запуск:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.chunking import split_documents
from rag.config import Settings
from rag.embeddings import HashingEmbeddings
from rag.loaders import load_document


class LoaderAndSplitterTests(unittest.TestCase):
    def test_loads_txt_and_splits(self) -> None:
        path = Path(__file__).resolve().parent.parent / "data" / "company.txt"
        document = load_document(path)
        self.assertIn("TechNova", document.page_content)
        self.assertEqual(document.metadata["file_type"], "txt")

        chunks = split_documents([document], chunk_size=500, chunk_overlap=50)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertTrue(chunk.page_content.strip())
            self.assertIn("chunk_index", chunk.metadata)
            self.assertLessEqual(len(chunk.page_content), 500)

    def test_hashing_embeddings_are_deterministic_and_normalized(self) -> None:
        embedder = HashingEmbeddings(dimensions=256)
        first = embedder.embed_query("Техническая поддержка TechNova")
        second = embedder.embed_query("Техническая поддержка TechNova")
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0, places=5)

        other = embedder.embed_query("совершенно другой текст про погоду")
        self.assertNotEqual(first, other)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.mkdtemp(prefix="rag-test-")
        cls.env = {
            "RAG_DATA_DIR": str(Path(__file__).resolve().parent.parent / "data"),
            "RAG_CHROMA_DIR": cls.temp_dir,
            "EMBEDDING_PROVIDER": "hashing",
            "LLM_PROVIDER": "extractive",
            "RAG_MIN_RELEVANCE": "0",
            "RAG_TOP_K": "3",
        }
        with patch.dict(os.environ, cls.env, clear=False):
            settings = Settings.from_env()
            from rag.chain import RagPipeline

            cls.pipeline = RagPipeline.build(settings, rebuild=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_answers_question_present_in_documents(self) -> None:
        result = self.pipeline.ask("В каком городе находится главный офис TechNova?")
        self.assertFalse(result.refused)
        self.assertIn("Казань", result.answer)
        self.assertTrue(result.sources)
        self.assertLessEqual(len(result.sources), 3)
        self.assertGreater(result.sources[0].score, 0.0)

    def test_answers_about_subscription_price(self) -> None:
        result = self.pipeline.ask("Сколько стоит расширенная подписка на TechNova CRM?")
        self.assertFalse(result.refused)
        self.assertTrue("5 000" in result.answer or "5000" in result.answer)

    def test_refuses_question_absent_from_documents(self) -> None:
        result = self.pipeline.ask("Кто генеральный директор TechNova?")
        self.assertTrue(result.refused)
        self.assertIn("нет информации", result.answer)


if __name__ == "__main__":
    unittest.main()
