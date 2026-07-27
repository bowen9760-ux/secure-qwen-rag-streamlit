"""项目配置。

所有项目内路径都以本文件所在目录为基准，避免从不同工作目录启动时读写到
意外位置。旧的 ``rag`` collection 不会被自动读取；v2 使用独立 collection。
"""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIRECTORY = PROJECT_ROOT / "data"

# Chroma 配置。rag_v2 与旧 rag collection 物理共存但逻辑隔离，便于安全回滚。
collection_name = os.getenv("RAG_COLLECTION_NAME", "rag_v2")
legacy_collection_name = "rag"
persist_directory = str(PROJECT_ROOT / "chroma_db")
data_directory = str(DATA_DIRECTORY)

# 文本分割参数。每个元素必须是独立字符串；最后的空串是递归切分兜底。
chunk_size = 1000
chunk_overlap = 100
separators = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    ".",
    "!",
    "?",
    ";",
    ",",
    " ",
    "",
]
max_spliter = chunk_size  # 保留旧配置名，供已有调用兼容

# 检索参数：先取 top-k，再按归一化 relevance score（越大越相关）过滤。
retrieval_k = int(os.getenv("RAG_RETRIEVAL_K", "4"))
similarity_score_threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.50"))
relevance_score_threshold = similarity_score_threshold

# 入库并发锁。
knowledge_lock_timeout_seconds = 60.0
max_document_bytes = 2 * 1024 * 1024
max_source_name_chars = 240

# 模型配置。
dashscope_base_url = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).rstrip("/")
embedding_model_name = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-v4")
embedding_dimensions = 1024
# 默认使用 Qwen Plus；生产环境可通过环境变量固定到经回归验证的日期快照。
chat_model_name = os.getenv("RAG_CHAT_MODEL", "qwen3.7-plus")
model_timeout_seconds = 60.0
model_max_retries = 2

# 会话默认只保存在当前进程内；只有明确配置后才落盘。
history_max_messages = 12
history_persist = os.getenv("RAG_HISTORY_PERSIST", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
history_directory = str(PROJECT_ROOT / "user_chat_history_v2")
history_retention_days = int(os.getenv("RAG_HISTORY_RETENTION_DAYS", "7"))
max_question_chars = 1000
