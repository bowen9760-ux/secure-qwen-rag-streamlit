"""模型客户端工厂。

阿里云百炼同时提供 OpenAI 兼容的聊天与向量接口。集中在这里创建客户端，
可以保证入库和检索始终使用相同的模型、端点和向量维度。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config_data as config


def _api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY")
    return api_key


def create_embeddings(model_name: str | None = None) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=model_name or config.embedding_model_name,
        dimensions=config.embedding_dimensions,
        api_key=_api_key(),
        base_url=config.dashscope_base_url,
        check_embedding_ctx_length=False,
        chunk_size=10,
        model_kwargs={"encoding_format": "float"},
        max_retries=config.model_max_retries,
        request_timeout=config.model_timeout_seconds,
    )


def create_chat_model(model_name: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or config.chat_model_name,
        api_key=_api_key(),
        base_url=config.dashscope_base_url,
        temperature=0,
        max_retries=config.model_max_retries,
        timeout=config.model_timeout_seconds,
        stream_usage=False,
    )


def embedding_identity(
    embedding: Any,
    model_name: str | None = None,
) -> dict[str, Any]:
    """返回可写入 Chroma 元数据的稳定嵌入配置身份。"""

    resolved_model = str(
        model_name or getattr(embedding, "model", None) or embedding.__class__.__name__
    )
    dimensions = getattr(embedding, "dimensions", None)
    endpoint = getattr(embedding, "openai_api_base", None)
    provider = f"{embedding.__class__.__module__}.{embedding.__class__.__qualname__}"
    payload = {
        "provider": provider,
        "model": resolved_model,
        "dimensions": int(dimensions) if dimensions is not None else None,
        "endpoint": str(endpoint or ""),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


__all__ = [
    "create_chat_model",
    "create_embeddings",
    "embedding_identity",
]
