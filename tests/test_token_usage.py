import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api import main
from api.main import normalize_usage


class TokenUsageTests(unittest.TestCase):
    def test_accepts_input_output_token_aliases(self):
        usage = normalize_usage({"input_tokens": 12, "output_tokens": 7})

        self.assertEqual(usage["prompt_tokens"], 12)
        self.assertEqual(usage["completion_tokens"], 7)
        self.assertEqual(usage["total_tokens"], 19)

    def test_estimates_missing_usage_from_request_and_response(self):
        usage = normalize_usage(
            {},
            [{"role": "user", "content": "Count these tokens"}],
            "A generated response",
        )

        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertGreater(usage["completion_tokens"], 0)
        self.assertEqual(
            usage["total_tokens"],
            usage["prompt_tokens"] + usage["completion_tokens"],
        )

    def test_replaces_zero_total_with_component_sum(self):
        usage = normalize_usage(
            {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 0}
        )

        self.assertEqual(usage["total_tokens"], 13)


class StreamingTokenUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_exposes_normalized_usage_for_request_log(self):
        class Source:
            def __init__(self):
                self.response = SimpleNamespace(aclose=AsyncMock())
                self.usage = None
                self.generated_text = "A generated response"

            def __aiter__(self):
                async def chunks():
                    yield b"data: chunk\n\n"

                return chunks()

        source = Source()
        principal = SimpleNamespace()
        body = {"messages": [{"role": "user", "content": "Count these tokens"}]}

        with (
            patch.object(main, "add_global_usage_async", new=AsyncMock()),
            patch.object(main, "record_client_completion_async", new=AsyncMock()),
        ):
            chunks = [
                chunk
                async for chunk in main.tracked_external_stream(
                    source, principal, "test-model", body
                )
            ]

        self.assertEqual(chunks, [b"data: chunk\n\n"])
        self.assertGreater(source.usage["prompt_tokens"], 0)
        self.assertGreater(source.usage["completion_tokens"], 0)
        self.assertEqual(
            source.usage["total_tokens"],
            source.usage["prompt_tokens"] + source.usage["completion_tokens"],
        )
        source.response.aclose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
