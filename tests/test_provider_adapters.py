import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from api.provider_adapters import AgnesAdapter, GroqAdapter
from api import agnes_provider
from api.agnes_provider import AsyncAgnesCandidatePool
from api.distributed_limits import admit_fixed_window


async def response_from_transport(content: bytes, content_type: str = "application/json"):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=content, headers={"content-type": content_type}
        )
    )
    client = httpx.AsyncClient(transport=transport)
    request = client.build_request("POST", "https://provider.test/v1")
    response = await client.send(request, stream=True)
    return client, response


class ProviderAdapterContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_groq_result_normalizes_json_and_records_pool_success(self):
        payload = {
            "choices": [{"finish_reason": "other"}],
            "usage": {"total_tokens": 17},
        }
        client, response = await response_from_transport(json.dumps(payload).encode())
        executor = AsyncMock(return_value=(response, "groq-model", "secret-key"))
        pool = SimpleNamespace(
            record_success=AsyncMock(), save_to_upstash=AsyncMock()
        )
        adapter = GroqAdapter(
            executor=executor,
            pool=pool,
            redis_url="https://redis.test",
            redis_token="token",
        )

        result = await adapter.chat(client, {"model": "groq-model"}, True)
        self.assertEqual(result.provider, "groq")
        self.assertEqual(result.model, "groq-model")
        self.assertEqual(result.credential_suffix, "-key")
        self.assertEqual(result.json()["choices"][0]["finish_reason"], "stop")

        await result.record_success(17)
        pool.record_success.assert_awaited_once_with("secret-key", "groq-model", 17)
        pool.save_to_upstash.assert_awaited_once_with(
            client, "https://redis.test", "token"
        )
        await result.close()
        self.assertTrue(response.is_closed)
        await client.aclose()

    async def test_groq_result_normalizes_stream_without_exposing_executor_tuple(self):
        body = b'data: {"choices":[{"finish_reason":"other"}]}\n\ndata: [DONE]\n\n'
        client, response = await response_from_transport(body, "text/event-stream")
        executor = AsyncMock(return_value=(response, "groq-model", "key-1234"))
        pool = SimpleNamespace(
            record_success=AsyncMock(), save_to_upstash=AsyncMock()
        )
        result = await GroqAdapter(executor=executor, pool=pool).chat(
            client, {"model": "groq-model", "stream": True}
        )

        chunks = [chunk async for chunk in result.stream("request-1")]
        output = b"".join(chunks).decode()
        self.assertIn('"finish_reason": "stop"', output)
        self.assertIn('"id": "chatcmpl-request-1"', output)
        self.assertIn('"model": "groq-model"', output)
        await result.close()
        await client.aclose()

    async def test_agnes_chat_result_owns_metadata_and_normalization(self):
        payload = {"model": "agnes-2.5-flash", "choices": [{"finish_reason": "other"}]}
        client, response = await response_from_transport(json.dumps(payload).encode())
        candidate = SimpleNamespace(site="cn", key="agnes-secret", affinity_id="cn:abc")
        executor = AsyncMock(return_value=(response, candidate, None))
        adapter = AgnesAdapter(chat_executor=executor)

        result = await adapter.chat(
            client, {"model": "agnes/agnes-2.5-flash"}, timeout=120.0
        )
        normalized = result.json("request-2")
        self.assertEqual(result.provider, "agnes")
        self.assertEqual(result.site, "cn")
        self.assertEqual(result.credential_suffix, "cret")
        self.assertEqual(result.route_id, "cn:abc")
        self.assertEqual(normalized["id"], "request-2")
        self.assertEqual(normalized["model"], "agnes/agnes-2.5-flash")
        self.assertEqual(normalized["choices"][0]["finish_reason"], "stop")
        executor.assert_awaited_once_with(
            client,
            {"model": "agnes/agnes-2.5-flash"},
            timeout=120.0,
        )
        await result.close()
        await client.aclose()

    async def test_agnes_media_result_preserves_route_and_public_model(self):
        payload = {"id": "video-1", "model": "agnes-video-v2.0"}
        client, response = await response_from_transport(json.dumps(payload).encode())
        candidate = SimpleNamespace(site="intl", key="key-9876", affinity_id="intl:def")
        executor = AsyncMock(return_value=(response, candidate, None))
        adapter = AgnesAdapter(request_executor=executor)

        result = await adapter.request(
            client,
            "POST",
            "/videos",
            body={"model": "agnes/agnes-video-v2.0"},
            public_model="agnes/agnes-video-v2.0",
            timeout=120.0,
        )
        self.assertEqual(result.route_id, "intl:def")
        self.assertEqual(result.json()["model"], "agnes/agnes-video-v2.0")
        await result.close()
        await client.aclose()

    async def test_agnes_affinity_is_stable_when_key_order_changes(self):
        first = AsyncAgnesCandidatePool([
            ("cn", "https://cn/v1", ["key-a", "key-b"]),
            ("intl", "https://intl/v1", ["key-c"]),
        ])
        second = AsyncAgnesCandidatePool([
            ("intl", "https://intl/v1", ["key-c"]),
            ("cn", "https://cn/v1", ["key-b", "key-a"]),
        ])
        first_id = next(c.affinity_id for c in first.candidates if c.key == "key-a")
        second_id = next(c.affinity_id for c in second.candidates if c.key == "key-a")
        self.assertEqual(first_id, second_id)

    async def test_distributed_limit_rejects_count_over_limit(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json=[{"result": 6}, {"result": 1}], request=request
            )
        )
        client = httpx.AsyncClient(transport=transport)
        admitted = await admit_fixed_window(
            client, "https://redis.test", "token", "provider", "key", "model", 5
        )
        self.assertFalse(admitted)
        await client.aclose()

    async def test_distributed_limit_returns_indeterminate_on_redis_failure(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(503, request=request)
        )
        client = httpx.AsyncClient(transport=transport)
        admitted = await admit_fixed_window(
            client, "https://redis.test", "token", "provider", "key", "model", 5
        )
        self.assertIsNone(admitted)
        await client.aclose()

    async def test_legacy_candidate_index_queries_original_video_credential(self):
        original_pool = agnes_provider.agnes_pool
        pool = AsyncAgnesCandidatePool([
            ("cn", "https://cn/v1", ["key-a", "key-b"]),
        ])
        agnes_provider.agnes_pool = pool
        seen = []

        def handler(request):
            seen.append(request.headers.get("Authorization"))
            return httpx.Response(200, json={"status": "completed"}, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            response, candidate, error = await agnes_provider.query_agnes_video_result(
                client, "task-1", candidate_index=1
            )
            self.assertIsNone(error)
            self.assertEqual(candidate.key, "key-b")
            self.assertEqual(seen, ["Bearer key-b"])
            await response.aclose()
        finally:
            agnes_provider.agnes_pool = original_pool
            await client.aclose()


if __name__ == "__main__":
    unittest.main()
