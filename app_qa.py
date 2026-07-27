from __future__ import annotations

import logging
import uuid
from typing import Any

import streamlit as st

import config_data as config
from provider_errors import describe_provider_error


WELCOME_MESSAGE = "你好！我是智能客服。请告诉我你想了解的问题。"
logger = logging.getLogger(__name__)


def _new_messages() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": WELCOME_MESSAGE}]


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk

    content = getattr(chunk, "content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def _initialize_service() -> None:
    try:
        from rag import RagService

        st.session_state["rag_service"] = RagService()
        st.session_state.pop("rag_service_error", None)
    except Exception:
        logger.exception("RAG service initialization failed")
        st.session_state["rag_service"] = None
        st.session_state["rag_service_error"] = (
            "问答服务初始化失败。请检查依赖、DASHSCOPE_API_KEY 和知识库配置。"
        )


st.set_page_config(page_title="RAG 智能客服", page_icon="💬")
st.title("RAG 智能客服")
st.caption("回答将优先依据已审核并入库的资料。")

if "rag_session_id" not in st.session_state:
    st.session_state["rag_session_id"] = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state["messages"] = _new_messages()
if "rag_service" not in st.session_state:
    _initialize_service()

session_id = st.session_state["rag_session_id"]
rag_service = st.session_state["rag_service"]
session_config = {"configurable": {"session_id": session_id}}

with st.sidebar:
    st.subheader("当前会话")
    st.caption(f"会话标识：{session_id[:8]}…")
    st.caption("每个浏览器会话使用独立标识，不与其他访客共享对话。")

    if st.button(
        "清空当前对话",
        use_container_width=True,
        disabled=rag_service is None,
    ):
        try:
            rag_service.clear_history(session_id)
            st.session_state["messages"] = _new_messages()
            st.rerun()
        except Exception:
            logger.exception("Failed to clear chat history")
            st.error("清空失败，请稍后重试。")

if rag_service is None:
    st.error(st.session_state["rag_service_error"])
    if st.button("重试连接", type="primary"):
        _initialize_service()
        st.rerun()
    st.stop()

st.divider()

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("请输入你的问题")
if prompt:
    if len(prompt) > config.max_question_chars:
        st.error(f"问题不能超过 {config.max_question_chars} 个字符。")
        st.stop()
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("正在检索资料并生成回答…"):
                response_stream = rag_service.conversation_chain.stream(
                    {"input": prompt},
                    session_config,
                )
                for chunk in response_stream:
                    text = _chunk_text(chunk)
                    if not text:
                        continue
                    full_response += text
                    response_placeholder.markdown(f"{full_response}▌")

            if not full_response.strip():
                raise RuntimeError("模型未返回有效内容")

            # 只更新当前聊天气泡中的占位符，避免把同一答案再渲染一次。
            response_placeholder.markdown(full_response)
            st.session_state["messages"].append(
                {"role": "assistant", "content": full_response}
            )
        except Exception as exc:
            provider_error = describe_provider_error(exc)
            if provider_error is None:
                logger.exception("RAG response generation failed")
                error_message = (
                    "回答生成失败，请稍后重试；当前问题不会被保存为完整回答。"
                )
            else:
                logger.warning(
                    "Model provider request failed: category=%s status=%s "
                    "code=%s request_id=%s",
                    provider_error.category,
                    provider_error.status_code,
                    provider_error.code,
                    provider_error.request_id,
                )
                error_message = provider_error.user_message

            if full_response:
                response_placeholder.markdown(full_response)
            else:
                response_placeholder.empty()
            st.error(error_message)
            if (
                provider_error is not None
                and provider_error.category == "free_tier_exhausted"
            ):
                st.link_button(
                    "打开百炼免费额度页面",
                    "https://bailian.console.aliyun.com/?tab=costing-balance",
                )
