"""将模型供应商异常转换为安全、可操作的用户提示。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)


_FREE_TIER_ONLY_CODE = "AllocationQuota.FreeTierOnly"


@dataclass(frozen=True, slots=True)
class ProviderErrorInfo:
    """可安全展示和记录的模型服务错误摘要。"""

    category: str
    user_message: str
    status_code: int | None
    code: str | None
    request_id: str | None


def _non_empty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _error_details(error: BaseException) -> dict[str, Any]:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return {}
    nested = body.get("error")
    return nested if isinstance(nested, dict) else body


def describe_provider_error(error: BaseException) -> ProviderErrorInfo | None:
    """识别常见 OpenAI 兼容接口错误，不向页面泄露原始响应正文。"""

    details = _error_details(error)
    status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int):
        status_code = None

    code = _non_empty_string(getattr(error, "code", None))
    error_type = _non_empty_string(getattr(error, "type", None))
    code = code or _non_empty_string(details.get("code"))
    error_type = error_type or _non_empty_string(details.get("type"))
    request_id = _non_empty_string(getattr(error, "request_id", None))

    if isinstance(error, PermissionDeniedError):
        if _FREE_TIER_ONLY_CODE in {code, error_type}:
            return ProviderErrorInfo(
                category="free_tier_exhausted",
                user_message=(
                    "当前模型的免费额度已用完，百炼已拒绝本次调用。"
                    "请在百炼控制台检查该模型额度；如需继续使用，请完成认证和充值，"
                    "并关闭“免费额度用完即停（仅使用免费额度）”后重试。"
                    "关闭后会产生按量费用。"
                ),
                status_code=status_code,
                code=code or error_type,
                request_id=request_id,
            )
        return ProviderErrorInfo(
            category="permission",
            user_message=(
                "模型服务拒绝访问。请检查 API Key 所属地域、业务空间和模型调用权限。"
            ),
            status_code=status_code,
            code=code or error_type,
            request_id=request_id,
        )

    if isinstance(error, AuthenticationError):
        return ProviderErrorInfo(
            category="authentication",
            user_message="模型服务鉴权失败。请检查 DASHSCOPE_API_KEY 是否正确且仍有效。",
            status_code=status_code,
            code=code or error_type,
            request_id=request_id,
        )

    if isinstance(error, RateLimitError):
        return ProviderErrorInfo(
            category="rate_limit",
            user_message="模型服务当前请求过多或配额受限，请稍后重试并检查模型用量。",
            status_code=status_code,
            code=code or error_type,
            request_id=request_id,
        )

    if isinstance(error, APITimeoutError):
        return ProviderErrorInfo(
            category="timeout",
            user_message="连接模型服务超时，请检查网络后重试。",
            status_code=status_code,
            code=code or error_type,
            request_id=request_id,
        )

    if isinstance(error, APIConnectionError):
        return ProviderErrorInfo(
            category="connection",
            user_message="无法连接模型服务，请检查网络和 DASHSCOPE_BASE_URL 后重试。",
            status_code=status_code,
            code=code or error_type,
            request_id=request_id,
        )

    return None


def should_abort_query_rewrite(error: BaseException) -> bool:
    """鉴权和权限问题重试下一阶段没有意义，应立即交给界面处理。"""

    info = describe_provider_error(error)
    return info is not None and info.category in {
        "authentication",
        "free_tier_exhausted",
        "permission",
    }


__all__ = [
    "ProviderErrorInfo",
    "describe_provider_error",
    "should_abort_query_rewrite",
]
