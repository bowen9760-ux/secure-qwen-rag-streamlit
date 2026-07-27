from __future__ import annotations

import hmac
import logging
import os
from pathlib import Path

import streamlit as st

import config_data as config


BASE_DIR = Path(__file__).resolve().parent
DATA_DIRECTORY = BASE_DIR / "data"
MAX_FILE_SIZE_BYTES = config.max_document_bytes
logger = logging.getLogger(__name__)


def _initialize_service() -> None:
    try:
        from knowledge_base import KnowledgeBase

        st.session_state["kb_service"] = KnowledgeBase()
        st.session_state.pop("kb_service_error", None)
    except Exception:
        logger.exception("Knowledge base service initialization failed")
        st.session_state["kb_service"] = None
        st.session_state["kb_service_error"] = (
            "知识库服务初始化失败。请检查依赖、DASHSCOPE_API_KEY 和存储配置。"
        )


def _is_authorized(provided_token: str, expected_token: str) -> bool:
    return hmac.compare_digest(
        provided_token.encode("utf-8"),
        expected_token.encode("utf-8"),
    )


st.set_page_config(page_title="RAG 知识库管理", page_icon="🔐")
st.title("RAG 知识库管理")
st.caption("仅管理员可上传资料或从 data/ 目录重建知识库。")

expected_admin_token = os.getenv("RAG_ADMIN_TOKEN", "")
provided_admin_token = st.text_input(
    "管理员令牌",
    type="password",
    autocomplete="current-password",
    help="令牌由服务端环境变量 RAG_ADMIN_TOKEN 配置。",
)

if not expected_admin_token:
    st.error("服务端尚未配置 RAG_ADMIN_TOKEN，管理功能已锁定。")
    st.stop()
if len(expected_admin_token) < 24:
    st.error("RAG_ADMIN_TOKEN 长度不足 24 个字符，管理功能已锁定。")
    st.stop()
if not provided_admin_token:
    st.info("请输入管理员令牌后继续。")
    st.stop()
if not _is_authorized(provided_admin_token, expected_admin_token):
    st.error("管理员令牌无效。")
    st.stop()

st.success("管理员身份已验证。")

if "kb_service" not in st.session_state:
    _initialize_service()

kb_service = st.session_state["kb_service"]
if kb_service is None:
    st.error(st.session_state["kb_service_error"])
    if st.button("重试连接", type="primary"):
        _initialize_service()
        st.rerun()
    st.stop()

st.divider()
st.subheader("上传单个文本文件")
st.caption("仅接受 UTF-8 编码的 .txt 文件，单个文件最大 2 MiB。")

with st.form("knowledge_upload_form", clear_on_submit=False):
    uploaded_file = st.file_uploader(
        "选择知识库文件",
        type=["txt"],
        accept_multiple_files=False,
        max_upload_size=max(1, MAX_FILE_SIZE_BYTES // (1024 * 1024)),
    )
    upload_submitted = st.form_submit_button(
        "确认上传并入库",
        type="primary",
        use_container_width=True,
    )

if upload_submitted:
    if uploaded_file is None:
        st.warning("请先选择一个 .txt 文件。")
    else:
        raw_content = uploaded_file.getvalue()
        safe_filename = Path(uploaded_file.name).name

        if len(raw_content) > MAX_FILE_SIZE_BYTES:
            st.error("文件超过 2 MiB 限制，请拆分后再上传。")
        elif not raw_content:
            st.error("不能上传空文件。")
        elif Path(safe_filename).suffix.lower() != ".txt":
            st.error("仅支持 .txt 文件。")
        else:
            try:
                file_content = raw_content.decode("utf-8-sig")
            except UnicodeDecodeError:
                st.error("文件不是有效的 UTF-8 编码，请转换编码后重试。")
            else:
                if not file_content.strip():
                    st.error("文件为空或仅包含空白字符。")
                elif "\x00" in file_content:
                    st.error("文件包含二进制空字符，不能作为文本知识入库。")
                else:
                    try:
                        with st.spinner("正在切分、向量化并写入知识库…"):
                            result = kb_service.upload_by_str(
                                file_content,
                                safe_filename,
                                operator="streamlit-admin",
                            )
                        st.success(str(result or "文件已成功入库。"))
                    except Exception:
                        logger.exception("Knowledge file upload failed")
                        st.error("文件入库失败，请检查模型服务与存储状态后重试。")

st.divider()
st.subheader("从 data/ 重建知识库")
st.warning(
    "重建会替换当前 rag_v2 集合中的内容。请先确认 data/ 目录只包含已审核的 UTF-8 文本。"
)
rebuild_confirmed = st.checkbox(
    "我确认要使用 data/ 目录重建当前知识库",
    key="rebuild_confirmed",
)

if st.button(
    "确认重建 rag_v2",
    disabled=not rebuild_confirmed,
    use_container_width=True,
):
    try:
        with st.spinner("正在重建知识库，请勿重复提交…"):
            result = kb_service.rebuild_from_directory(str(DATA_DIRECTORY))
        st.success(str(result or "知识库重建完成。"))
    except Exception:
        logger.exception("Knowledge base rebuild failed")
        st.error("知识库重建失败；服务会尽量保留重建前的可用数据。")
