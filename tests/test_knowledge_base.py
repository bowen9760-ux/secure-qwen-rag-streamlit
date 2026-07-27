import hashlib
import os
import tempfile
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_core.embeddings import Embeddings

from knowledge_base import KnowledgeBase
from vector_stores import VectorStore


class HashEmbeddings(Embeddings):
    """稳定、离线且无需模型服务的测试向量。"""

    dimensions = 32
    model = "offline-hash-v1"

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [(byte - 127.5) / 127.5 for byte in digest[: cls.dimensions]]
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ChangedHashEmbeddings(HashEmbeddings):
    model = "offline-hash-v2"


class KnowledgeBaseIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.persist_directory = Path(self.temp_directory.name) / "chroma"
        self.collection_name = f"test_{uuid.uuid4().hex}"
        self.embedding = HashEmbeddings()
        self.knowledge_base = KnowledgeBase(
            embedding=self.embedding,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_model_name="offline-hash-v1",
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def _records(self, source_key: str):
        return self.knowledge_base.chroma.get(
            where={"source_key": source_key},
            include=["documents", "metadatas"],
        )

    def test_large_document_is_split_and_same_source_is_replaced(self):
        large_text = "\n\n".join(
            f"段落 {index}：" + ("甲乙丙丁" * 60) for index in range(12)
        )
        result = self.knowledge_base.upload_by_str(large_text, "guide.txt")
        self.assertEqual(result, "文件上传成功")
        old_records = self._records("guide.txt")
        self.assertGreater(len(old_records["ids"]), 1)

        replacement = "这是同名文件的新版本。"
        result = self.knowledge_base.upload_by_str(replacement, "guide.txt")
        self.assertIn("旧版本已替换", result)
        new_records = self._records("guide.txt")
        self.assertEqual(len(new_records["ids"]), 1)
        self.assertEqual(new_records["documents"], [replacement])

    def test_same_content_from_different_sources_is_traceable(self):
        content = "相同正文可以来自两个独立且可审计的来源。"
        self.knowledge_base.upload_by_str(content, "a.txt")
        self.knowledge_base.upload_by_str(content, "b.txt")
        self.assertEqual(self.knowledge_base.stats()["documents"], 2)

    def test_source_path_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            self.knowledge_base.upload_by_str("正文", "../escape.txt")

    def test_mixed_embedding_spaces_are_rejected(self):
        self.knowledge_base.upload_by_str("当前向量空间", "space.txt")
        with self.assertRaises(RuntimeError):
            KnowledgeBase(
                embedding=ChangedHashEmbeddings(),
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )

    def test_concurrent_same_source_updates_converge_to_one_version(self):
        contents = ["并发版本 A", "并发版本 B"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(
                executor.map(
                    lambda content: self.knowledge_base.upload_by_str(
                        content,
                        "concurrent.txt",
                    ),
                    contents,
                )
            )
        records = self._records("concurrent.txt")
        self.assertEqual(len(records["ids"]), 1)
        self.assertIn(records["documents"][0], contents)

    def test_directory_rebuild_removes_deleted_sources(self):
        source_directory = Path(self.temp_directory.name) / "source"
        source_directory.mkdir()
        (source_directory / "a.txt").write_text("A 文档", encoding="utf-8")
        (source_directory / "b.txt").write_text("B 文档", encoding="utf-8")
        first = self.knowledge_base.rebuild_from_directory(source_directory)
        self.assertEqual(first["files"], 2)
        self.assertEqual(self.knowledge_base.stats()["documents"], 2)

        (source_directory / "b.txt").unlink()
        second = self.knowledge_base.rebuild_from_directory(source_directory)
        self.assertGreaterEqual(second["deleted_chunks"], 1)
        self.assertEqual(self.knowledge_base.stats()["documents"], 1)

    def test_scored_search_contract_matches_rag_service(self):
        content = "精确检索测试文本"
        self.knowledge_base.upload_by_str(content, "search.txt")
        vector_store = VectorStore(
            embedding=self.embedding,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        results = vector_store.similarity_search_with_relevance_scores(
            content,
            k=4,
            score_threshold=0.0,
        )
        self.assertTrue(results)
        self.assertEqual(results[0][0].metadata["source"], "search.txt")
        self.assertGreaterEqual(results[0][1], 0.99)
        documents = vector_store.get_retriever(
            k=4,
            score_threshold=0.0,
        ).invoke(content)
        self.assertEqual(documents[0].metadata["source"], "search.txt")


if __name__ == "__main__":
    unittest.main()
