"""Chroma 向量检索封装。"""

from pathlib import Path
from typing import Any

from filelock import FileLock
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

import config_data as config
from model_clients import embedding_identity


class VectorStore:
    """连接指定 Chroma collection，并提供带真实分数阈值的检索器。

    ``embedding``、``persist_directory`` 和 ``collection_name`` 都可注入，
    因而测试时可使用 FakeEmbeddings 和临时目录，不会触发外部模型调用。
    """

    def __init__(
        self,
        embedding: Any,
        persist_directory: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        if embedding is None:
            raise ValueError("embedding 不能为空")

        self.embedding = embedding
        self.persist_directory = str(
            Path(persist_directory or config.persist_directory).expanduser().resolve()
        )
        self.collection_name = collection_name or config.collection_name
        self._knowledge_lock = FileLock(
            str(Path(self.persist_directory) / f".{self.collection_name}.write.lock"),
            timeout=config.knowledge_lock_timeout_seconds,
        )
        self.embedding_identity = embedding_identity(self.embedding)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embedding,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._assert_embedding_compatible()

    def _assert_embedding_compatible(self) -> None:
        records = self.vector_store.get(include=["metadatas"])
        metadatas = [metadata or {} for metadata in records.get("metadatas") or []]
        if not metadatas:
            return
        fingerprints = {metadata.get("embedding_fingerprint") for metadata in metadatas}
        if fingerprints != {self.embedding_identity["fingerprint"]}:
            raise RuntimeError(
                "向量库与查询嵌入配置不一致；请切换到匹配的 collection。"
            )

    def get_retriever(
        self,
        k: int | None = None,
        score_threshold: float | None = None,
    ):
        """返回先取 ``k`` 条、再按 relevance score 过滤的检索器。"""

        resolved_k = config.retrieval_k if k is None else k
        resolved_threshold = (
            config.similarity_score_threshold
            if score_threshold is None
            else score_threshold
        )
        if not isinstance(resolved_k, int) or resolved_k <= 0:
            raise ValueError("k 必须是正整数")
        if not 0.0 <= float(resolved_threshold) <= 1.0:
            raise ValueError("score_threshold 必须位于 [0, 1]")

        return RunnableLambda(
            lambda query: [
                document
                for document, _ in self.similarity_search_with_relevance_scores(
                    query,
                    k=resolved_k,
                    score_threshold=float(resolved_threshold),
                )
            ],
            name="chroma_score_threshold_retriever",
        )

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int | None = None,
        score_threshold: float | None = None,
    ):
        """返回 ``(Document, relevance_score)``，供需要展示来源的 RAG 链使用。"""

        if not isinstance(query, str) or not query.strip():
            return []
        resolved_k = config.retrieval_k if k is None else k
        resolved_threshold = (
            config.similarity_score_threshold
            if score_threshold is None
            else score_threshold
        )
        if not isinstance(resolved_k, int) or resolved_k <= 0:
            raise ValueError("k 必须是正整数")
        if not 0.0 <= float(resolved_threshold) <= 1.0:
            raise ValueError("score_threshold 必须位于 [0, 1]")

        # collection 明确使用 cosine；Chroma 返回距离（越小越相关），统一转换为
        # [0, 1] relevance score，避免不同 LangChain 版本的默认归一化漂移。
        with self._knowledge_lock:
            results = self.vector_store.similarity_search_with_score(
                query.strip(),
                k=resolved_k,
            )
        return [
            (document, relevance)
            for document, distance in results
            if (relevance := max(0.0, min(1.0, 1.0 - float(distance))))
            >= float(resolved_threshold)
        ]


if __name__ == "__main__":
    from model_clients import create_embeddings

    retriever = VectorStore(create_embeddings()).get_retriever()
    print(retriever.invoke("我的身高182，尺码推荐"))
