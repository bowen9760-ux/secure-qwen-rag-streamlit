import unittest

import httpx
from openai import PermissionDeniedError

from provider_errors import describe_provider_error, should_abort_query_rewrite


def _permission_error(
    *,
    code: str,
    nested_body: bool = False,
) -> PermissionDeniedError:
    request = httpx.Request(
        "POST",
        "https://example.invalid/v1/chat/completions",
    )
    response = httpx.Response(403, request=request)
    details = {
        "message": "provider detail must not be shown to users",
        "type": code,
        "code": code,
        "param": None,
    }
    body = {"error": details} if nested_body else details
    return PermissionDeniedError("request rejected", response=response, body=body)


class ProviderErrorTests(unittest.TestCase):
    def test_free_tier_error_has_actionable_safe_message(self):
        error = _permission_error(code="AllocationQuota.FreeTierOnly")

        info = describe_provider_error(error)

        self.assertIsNotNone(info)
        self.assertEqual(info.category, "free_tier_exhausted")
        self.assertIn("免费额度已用完", info.user_message)
        self.assertIn("关闭", info.user_message)
        self.assertIn("按量费用", info.user_message)
        self.assertNotIn("provider detail", info.user_message)
        self.assertTrue(should_abort_query_rewrite(error))

    def test_nested_provider_body_is_supported(self):
        error = _permission_error(
            code="AllocationQuota.FreeTierOnly",
            nested_body=True,
        )

        info = describe_provider_error(error)

        self.assertIsNotNone(info)
        self.assertEqual(info.category, "free_tier_exhausted")

    def test_other_permission_error_is_not_misreported_as_quota(self):
        error = _permission_error(code="AccessDenied")

        info = describe_provider_error(error)

        self.assertIsNotNone(info)
        self.assertEqual(info.category, "permission")
        self.assertNotIn("免费额度已用完", info.user_message)

    def test_unknown_error_is_left_for_generic_handler(self):
        self.assertIsNone(describe_provider_error(RuntimeError("internal detail")))


if __name__ == "__main__":
    unittest.main()
