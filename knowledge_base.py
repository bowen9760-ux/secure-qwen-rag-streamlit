"""可版本化、可审计的 Chroma 知识库入库服务。"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from filelock import FileLock
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
from model_clients import embedding_identity


def get_sha256(input_str: str) -> str:
    """返回 UTF-8 文本的 SHA-256。"""

    if not isinstance(input_str, str):
        raise TypeError("input_str 必须是字符串")
    return hashlib.sha256(input_str.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _normalize_source(filename: str) -> str:
    """把来源限制为短小、相对且无控制字符的 POSIX 风格标识。"""

    if not isinstance(filename, str):
        raise TypeError("filename 必须是字符串")
    normalized = unicodedata.normalize("NFKC", filename).strip().replace("\\", "/")
    if not normalized or any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise ValueError("filename 为空或包含控制字符")
    path = PurePosixPath(normalized)
    parts = path.parts
    has_windows_drive = bool(parts and len(parts[0]) == 2 and parts[0][1] == ":")
    if path.is_absolute() or has_windows_drive or ".." in parts:
        raise ValueError("filename 必须是知识库内的相对路径，不能包含路径穿越")
    source = path.as_posix()
    if source in {"", "."}:
        raise ValueError("filename 不能为空")
    if len(source) > config.max_source_name_chars:
        raise ValueError(f"filename 不能超过 {config.max_source_name_chars} 个字符")
    return source


def _source_key(filename: str) -> str:
    """生成稳定、大小写不敏感且不含本机路径分隔差异的来源键。"""

    return _normalize_source(filename).casefold()


def _normalize_operator(operator: Any) -> str:
    value = unicodedata.normalize("NFKC", str(operator or "unknown"))
    value = "".join(
        " " if ord(char) < 32 or ord(char) == 127 else char for char in value
    )
    return value.strip()[:100] or "unknown"


class KnowledgeBase:
    """知识库写入服务。

    同一 ``source_key`` 只保留一个当前版本。更新时先写入新分块，再删除旧分块，
    从而避免嵌入调用失败时先丢失线上旧版本。
    """

    schema_version = 2

    def __init__(
        self,
        embedding: Any | None = None,
        persist_directory: str | Path | None = None,
        collection_name: str | None = None,
        embedding_model_name: str | None = None,
    ) -> None:
        self.persist_directory = (
            Path(persist_directory or config.persist_directory).expanduser().resolve()
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or config.collection_name
        self.embedding_model_name = embedding_model_name or config.embedding_model_name

        if embedding is None:
            from model_clients import create_embeddings

            embedding = create_embeddings(self.embedding_model_name)
        elif embedding_model_name is None:
            self.embedding_model_name = str(
                getattr(embedding, "model", embedding.__class__.__name__)
            )
        self.embedding = embedding
        self.embedding_identity = embedding_identity(
            self.embedding,
            self.embedding_model_name,
        )

        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=str(self.persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._assert_embedding_compatible()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=list(config.separators),
            length_function=len,
        )
        self.spliter = self.splitter  # 兼容旧属性拼写
        self._splitter_fingerprint = get_sha256(
            json.dumps(
                {
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "separators": config.separators,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        self._lock_path = self.persist_directory / f".{self.collection_name}.write.lock"

    def _assert_embedding_compatible(self) -> None:
        """拒绝把不同嵌入空间静默混入同一 collection。"""

        records = self.chroma.get(include=["metadatas"])
        metadatas = [metadata or {} for metadata in records.get("metadatas") or []]
        if not metadatas:
            return
        fingerprints = {metadata.get("embedding_fingerprint") for metadata in metadatas}
        current = self.embedding_identity["fingerprint"]
        if fingerprints != {current}:
            raise RuntimeError(
                "当前 collection 的嵌入配置与运行配置不一致；"
                "请设置新的 RAG_COLLECTION_NAME 并重建，不能混用旧向量。"
            )

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        with FileLock(
            str(self._lock_path),
            timeout=config.knowledge_lock_timeout_seconds,
        ):
            yield

    def _split(self, data: str) -> list[str]:
        chunks = self.splitter.split_text(data)
        return chunks or [data]

    def _chunk_id(
        self,
        source_key: str,
        document_sha256: str,
        chunk_index: int,
        chunk: str,
    ) -> str:
        identity = "\x1f".join(
            (
                str(self.schema_version),
                self.collection_name,
                source_key,
                document_sha256,
                self._splitter_fingerprint,
                self.embedding_identity["fingerprint"],
                str(chunk_index),
                get_sha256(chunk),
            )
        )
        return get_sha256(identity)

    def _records_for_source(self, source_key: str, source: str) -> dict[str, Any]:
        records = self.chroma.get(
            where={"source_key": source_key},
            include=["metadatas"],
        )
        ids = list(records.get("ids") or [])
        metadatas = list(records.get("metadatas") or [])

        # 兼容可能由早期 v2 代码写入、尚无 source_key 的同名记录。
        legacy = self.chroma.get(where={"source": source}, include=["metadatas"])
        known = set(ids)
        for record_id, metadata in zip(
            legacy.get("ids") or [], legacy.get("metadatas") or []
        ):
            if record_id not in known:
                ids.append(record_id)
                metadatas.append(metadata)
                known.add(record_id)
        return {"ids": ids, "metadatas": metadatas}

    def _upload_locked(
        self,
        data: str,
        filename: str,
        operator: str,
    ) -> dict[str, Any]:
        if not isinstance(data, str):
            raise TypeError("data 必须是字符串")
        if not data.strip():
            raise ValueError("不能上传空文档")
        if "\x00" in data:
            raise ValueError("文档不能包含空字符")
        if len(data.encode("utf-8")) > config.max_document_bytes:
            raise ValueError(f"文档不能超过 {config.max_document_bytes} 字节")

        source = _normalize_source(filename)
        source_key = _source_key(source)
        document_sha256 = get_sha256(data)
        chunks = self._split(data)
        chunk_ids = [
            self._chunk_id(source_key, document_sha256, index, chunk)
            for index, chunk in enumerate(chunks)
        ]
        existing = self._records_for_source(source_key, source)
        existing_ids = set(existing["ids"])
        expected_ids = set(chunk_ids)
        obsolete_ids = sorted(existing_ids - expected_ids)

        # 上次若“新版本已写入、旧版本尚未删除”便在这里自动收敛。
        if expected_ids.issubset(existing_ids):
            if obsolete_ids:
                self.chroma.delete(ids=obsolete_ids)
            return {
                "status": "unchanged" if not obsolete_ids else "recovered",
                "source": source,
                "sha256": document_sha256,
                "chunks": len(chunks),
                "removed_chunks": len(obsolete_ids),
            }

        # 清理上次不完整写入的当前版本 ID；旧完整版本仍保留到新版本写成功。
        partial_ids = sorted(existing_ids & expected_ids)
        if partial_ids:
            self.chroma.delete(ids=partial_ids)

        ingested_at = _utc_now()
        ingestion_id = get_sha256(
            "\x1f".join(
                (
                    source_key,
                    document_sha256,
                    self._splitter_fingerprint,
                    self.embedding_identity["fingerprint"],
                )
            )
        )
        metadatas = []
        for index, chunk in enumerate(chunks):
            metadatas.append(
                {
                    "schema_version": self.schema_version,
                    "collection": self.collection_name,
                    "source": source,
                    "source_key": source_key,
                    "document_sha256": document_sha256,
                    "document_version": document_sha256[:12],
                    "chunk_sha256": get_sha256(chunk),
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "ingestion_id": ingestion_id,
                    "ingested_at": ingested_at,
                    "operator": _normalize_operator(operator),
                    "embedding_model": self.embedding_model_name,
                    "embedding_dimensions": (
                        self.embedding_identity["dimensions"] or 0
                    ),
                    "embedding_provider": self.embedding_identity["provider"],
                    "embedding_endpoint": self.embedding_identity["endpoint"],
                    "embedding_fingerprint": self.embedding_identity["fingerprint"],
                    "splitter_sha256": self._splitter_fingerprint,
                }
            )

        self.chroma.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=chunk_ids,
        )
        if obsolete_ids:
            self.chroma.delete(ids=obsolete_ids)
        return {
            "status": "created" if not existing_ids else "updated",
            "source": source,
            "sha256": document_sha256,
            "chunks": len(chunks),
            "removed_chunks": len(obsolete_ids),
        }

    def upload_by_str(
        self,
        data: str,
        filename: str,
        operator: str = "system",
    ) -> str:
        """写入或替换同名文档，保留原有的中文字符串返回值接口。"""

        with self._write_lock():
            result = self._upload_locked(data, filename, operator)

        messages = {
            "created": "文件上传成功",
            "updated": "文件已更新，旧版本已替换",
            "unchanged": "文件内容未变化，无需重复上传",
            "recovered": "文件内容未变化，已清理残留旧版本",
        }
        return messages[result["status"]]

    def rebuild_from_directory(
        self,
        directory: str | Path | None = None,
        operator: str = "rebuild",
        delete_missing: bool = True,
    ) -> dict[str, Any]:
        """将目录中的 ``*.txt`` 与 collection 同步。

        字符串路径完全支持。相对路径以当前工作目录解析；默认目录使用配置中的
        绝对 ``data_directory``。全部文件成功后才删除目录中已不存在的来源。
        """

        root = Path(directory or config.data_directory).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"知识目录不存在：{root}")
        files = sorted(path for path in root.rglob("*.txt") if path.is_file())
        summary: dict[str, Any] = {
            "directory": str(root),
            "files": len(files),
            "created": 0,
            "updated": 0,
            "unchanged": 0,
            "recovered": 0,
            "deleted_chunks": 0,
            "sources": [],
        }

        with self._write_lock():
            live_source_keys: set[str] = set()
            for path in files:
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError(f"知识目录不允许符号链接或越界文件：{path}")
                if path.stat().st_size > config.max_document_bytes:
                    raise ValueError(
                        f"知识文件超过 {config.max_document_bytes} 字节：{path}"
                    )
                source = path.relative_to(root).as_posix()
                data = path.read_text(encoding="utf-8-sig")
                if not data.strip():
                    raise ValueError(f"不能入库空文档：{path}")
                result = self._upload_locked(data, source, operator)
                live_source_keys.add(_source_key(source))
                summary[result["status"]] += 1
                summary["sources"].append(result)

            if delete_missing:
                all_records = self.chroma.get(include=["metadatas"])
                stale_ids = []
                for record_id, metadata in zip(
                    all_records.get("ids") or [],
                    all_records.get("metadatas") or [],
                ):
                    key = (metadata or {}).get("source_key")
                    if key not in live_source_keys:
                        stale_ids.append(record_id)
                if stale_ids:
                    self.chroma.delete(ids=stale_ids)
                summary["deleted_chunks"] = len(stale_ids)
        return summary

    def stats(self) -> dict[str, Any]:
        """返回 collection 的可审计概览，不调用嵌入或大模型 API。"""

        records = self.chroma.get(include=["metadatas"])
        metadatas = [metadata or {} for metadata in records.get("metadatas") or []]
        sources = {
            metadata.get("source_key")
            for metadata in metadatas
            if metadata.get("source_key")
        }
        versions = {
            (metadata.get("source_key"), metadata.get("document_sha256"))
            for metadata in metadatas
            if metadata.get("source_key") and metadata.get("document_sha256")
        }
        timestamps = [
            metadata["ingested_at"]
            for metadata in metadatas
            if metadata.get("ingested_at")
        ]
        return {
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "schema_version": self.schema_version,
            "chunks": len(records.get("ids") or []),
            "documents": len(sources),
            "versions": len(versions),
            "embedding_models": sorted(
                {
                    metadata["embedding_model"]
                    for metadata in metadatas
                    if metadata.get("embedding_model")
                }
            ),
            "embedding_fingerprints": sorted(
                {
                    metadata["embedding_fingerprint"]
                    for metadata in metadatas
                    if metadata.get("embedding_fingerprint")
                }
            ),
            "latest_ingested_at": max(timestamps) if timestamps else None,
        }


if __name__ == "__main__":
    try:
        service = KnowledgeBase()
        print(service.rebuild_from_directory(config.data_directory))
        print(service.stats())
    except RuntimeError as exc:
        raise SystemExit(f"配置或知识库错误：{exc}") from None
