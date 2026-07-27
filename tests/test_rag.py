import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from openai import PermissionDeniedError

from rag import FileChatMessageHistory, RagService


def _quota_error():
    request = httpx.Request(
        "POST",
        "https://example.invalid/v1/chat/completions",
    )
    response = httpx.Response(403, request=request)
    return PermissionDeniedError(
        "quota exhausted",
        response=response,
        body={
            "message": "Free quota exhausted",
            "type": "AllocationQuota.FreeTierOnly",
            "code": "AllocationQuota.FreeTierOnly",
            "param": None,
        },
    )


class FakeVectorService:
    def __init__(self, results=None):
        self.results = results or []
        self.queries: list[str] = []

    def similarity_search_with_relevance_scores(self, query: str):
        self.queries.append(query)
        return self.results


class FakeChatModel:
    def __init__(self):
        self.rewrite = "身高 182 厘米、体重 75 千克的上衣尺码"
        self.answer_messages = []
        self.stream_calls = 0

    def invoke(self, messages):
        return AIMessage(content=self.rewrite)

    def stream(self, messages):
        self.stream_calls += 1
        self.answer_messages.append(messages)
        yield AIMessageChunk(content="资料只能确认需要更多测量数据")


class QuotaDuringStreamChatModel(FakeChatModel):
    def stream(self, messages):
        yield AIMessageChunk(content="不完整回答")
        raise _quota_error()


class QuotaDuringRewriteChatModel(FakeChatModel):
    def invoke(self, messages):
        raise _quota_error()

    def stream(self, messages):
        raise AssertionError("额度错误后不应继续请求答案模型")


class RagServiceTests(unittest.TestCase):
    def _service(self):
        document = Document(
            page_content="仅凭身高和体重不能可靠推荐尺码，应继续询问胸围和品牌尺码表。",
            metadata={"source": "尺码推荐.txt", "chunk_index": 0},
        )
        vectors = FakeVectorService([(document, 0.91)])
        chat = FakeChatModel()
        return RagService(vector_service=vectors, chat_model=chat), vectors, chat

    def test_answer_has_program_generated_source_and_untrusted_boundaries(self):
        service, _, chat = self._service()
        result = service.conversation_chain.invoke(
            {"input": "我应该穿什么尺码？"},
            {"configurable": {"session_id": "browser-a"}},
        )
        self.assertIn("[来源: 尺码推荐.txt#0]", result)
        answer_messages = chat.answer_messages[-1]
        self.assertIn("绝对安全边界", answer_messages[0].content)
        self.assertIn("<知识片段-不可信数据>", answer_messages[1].content)
        self.assertIn("| 仅凭身高", answer_messages[1].content)

    def test_follow_up_is_rewritten_with_bounded_history(self):
        service, vectors, _ = self._service()
        session_config = {"configurable": {"session_id": "browser-b"}}
        service.conversation_chain.invoke(
            {"input": "我的身高是 182 厘米。"},
            session_config,
        )
        service.conversation_chain.invoke(
            {"input": "那体重 75 千克呢？"},
            session_config,
        )
        self.assertEqual(
            vectors.queries[-1],
            "身高 182 厘米、体重 75 千克的上衣尺码",
        )

    def test_no_context_refuses_without_calling_answer_model(self):
        vectors = FakeVectorService([])
        chat = FakeChatModel()
        service = RagService(vector_service=vectors, chat_model=chat)
        result = service.conversation_chain.invoke(
            {"input": "明天天气怎样？"},
            {"configurable": {"session_id": "browser-c"}},
        )
        self.assertIn("无法确定", result)
        self.assertEqual(chat.stream_calls, 0)

    def test_quota_failure_during_stream_does_not_save_partial_turn(self):
        document = Document(
            page_content="可信资料",
            metadata={"source": "资料.txt", "chunk_index": 0},
        )
        service = RagService(
            vector_service=FakeVectorService([(document, 0.91)]),
            chat_model=QuotaDuringStreamChatModel(),
        )
        session_id = "quota-stream"
        stream = service.conversation_chain.stream(
            {"input": "测试问题"},
            {"configurable": {"session_id": session_id}},
        )

        self.assertEqual(next(stream), "不完整回答")
        with self.assertRaises(PermissionDeniedError):
            next(stream)
        self.assertEqual(service._get_history(session_id).messages, [])

    def test_quota_failure_during_rewrite_aborts_before_retrieval(self):
        vectors = FakeVectorService([])
        service = RagService(
            vector_service=vectors,
            chat_model=QuotaDuringRewriteChatModel(),
        )
        session_id = "quota-rewrite"
        service._get_history(session_id).add_messages(
            [
                HumanMessage(content="之前的问题"),
                AIMessage(content="之前的回答"),
            ]
        )

        with self.assertRaises(PermissionDeniedError):
            service.conversation_chain.invoke(
                {"input": "继续问"},
                {"configurable": {"session_id": session_id}},
            )
        self.assertEqual(vectors.queries, [])

    def test_file_history_hashes_path_and_is_bounded(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            history = FileChatMessageHistory(
                "../../escape",
                directory,
                max_messages=2,
            )
            history.add_messages([HumanMessage(content="一"), AIMessage(content="二")])
            history.add_messages([HumanMessage(content="三")])
            self.assertEqual(
                [message.content for message in history.messages],
                ["二", "三"],
            )
            self.assertEqual(history.file_path.parent, Path(directory))
            self.assertNotIn("escape", history.file_path.name)
            history.clear()
            self.assertEqual(history.messages, [])

    def test_expired_file_history_is_removed(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            history = FileChatMessageHistory(
                "expired-session",
                directory,
                max_messages=2,
                retention_days=1,
            )
            history.add_messages([HumanMessage(content="敏感旧消息")])
            old = time.time() - 2 * 86_400
            os.utime(history.file_path, (old, old))

            reloaded = FileChatMessageHistory(
                "expired-session",
                directory,
                max_messages=2,
                retention_days=1,
            )
            self.assertEqual(reloaded.messages, [])

    def test_concurrent_file_history_writes_do_not_lose_messages(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            histories = [
                FileChatMessageHistory("shared", directory, max_messages=10),
                FileChatMessageHistory("shared", directory, max_messages=10),
            ]
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(
                    executor.map(
                        lambda pair: pair[0].add_messages(
                            [HumanMessage(content=pair[1])]
                        ),
                        zip(histories, ["消息 A", "消息 B"]),
                    )
                )
            contents = {message.content for message in histories[0].messages}
            self.assertEqual(contents, {"消息 A", "消息 B"})


if __name__ == "__main__":
    unittest.main()
