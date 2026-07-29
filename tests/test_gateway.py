import asyncio
import importlib
import os
import unittest

from fastapi.testclient import TestClient


os.environ["CEREBRAS_API_KEYS"] = "test-cerebras"
os.environ["GROQ_API_KEYS"] = "test-groq"
os.environ["AGNES_CN_API_KEYS"] = "test-agnes"
os.environ["ADMIN_API_KEY"] = "test-admin"
os.environ["CUSTOM_API_KEYS"] = ""
os.environ["GATEWAY_KEYS_JSON"] = (
    '{"contributor-a":{"name":"A","providers":'
    '{"cerebras":2,"groq":1,"agnes":0}}}'
)
os.environ.pop("UPSTASH_REDIS_REST_URL", None)
os.environ.pop("UPSTASH_REDIS_REST_TOKEN", None)

main = importlib.import_module("api.main")
access_control = importlib.import_module("api.access_control")
catalog = importlib.import_module("api.model_catalog")


class GatewaySecurityTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def admin_client(self):
        response = self.client.post(
            "/admin", data={"action": "login", "admin_key": "test-admin"}
        )
        self.assertEqual(response.status_code, 200)
        self.client.cookies.update(response.cookies)
        return response.cookies

    def test_management_pages_require_admin(self):
        for path in ["/status", "/log", "/debug", "/config", "/thinkingdisplay", "/fallbackmode"]:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 401)

    def test_control_get_does_not_mutate(self):
        cookies = self.admin_client()
        main.THINKING_MODE = "auto"
        response = self.client.get("/thinkingdisplay?mode=off")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(main.THINKING_MODE, "auto")

    def test_control_post_requires_csrf(self):
        cookies = self.admin_client()
        response = self.client.post(
            "/thinkingdisplay", data={"mode": "off", "csrf": "bad"}
        )
        self.assertEqual(response.status_code, 403)

    def test_control_post_changes_mode_with_csrf(self):
        cookies = self.admin_client()
        response = self.client.post(
            "/thinkingdisplay",
            data={"mode": "off", "csrf": main.admin_csrf_token(type("R", (), {"cookies": self.client.cookies})())},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(main.THINKING_MODE, "off")

    def test_debug_payload_capture_is_disabled_by_default(self):
        asyncio.run(main.add_debug_log_async({"request_body": "secret", "response_body": "secret"}))
        self.assertEqual(main.DEBUG_LOGS[0]["request_body"], "[payload capture disabled]")


class CatalogAndAccessTests(unittest.TestCase):
    def test_catalog_models_are_unique_and_classified(self):
        self.assertEqual(len(catalog.MODEL_CATALOG), len(set(catalog.MODEL_CATALOG)))
        for model_id, spec in catalog.MODEL_CATALOG.items():
            self.assertEqual(model_id, spec.public_id)
            self.assertIn(spec.provider, {"cerebras", "groq", "agnes"})
            self.assertIn(spec.operation, {"chat", "image", "video"})

    def test_contributor_provider_filtering(self):
        principal = asyncio.run(
            access_control.access_manager.authenticate("Bearer contributor-a")
        )
        self.assertTrue(access_control.access_manager.authorize(principal, "gpt-oss-120b"))
        self.assertTrue(
            access_control.access_manager.authorize(principal, "openai/gpt-oss-120b")
        )
        self.assertFalse(
            access_control.access_manager.authorize(principal, "agnes/agnes-2.5-flash")
        )

    def test_provider_quota_is_shared_across_models(self):
        manager = access_control.AccessManager()
        principal = asyncio.run(manager.authenticate("Bearer contributor-a"))
        first = asyncio.run(manager.consume(principal, "gpt-oss-120b", estimated_tokens=100))
        second = asyncio.run(manager.consume(principal, "zai-glm-4.7", estimated_tokens=100))
        self.assertEqual(first["current_rpm"], 1)
        self.assertEqual(second["current_rpm"], 2)
        self.assertEqual(second["limit_rpm"], 20)
        self.assertEqual(second["limit_tpm"], 120000)

    def test_legacy_policy_update_and_delete_removes_hashed_duplicate(self):
        manager = access_control.AccessManager()
        stored = {}

        async def setter(key, value):
            stored.clear()
            stored.update(value)
            return True

        principal = asyncio.run(manager.authenticate("Bearer contributor-a"))
        self.assertTrue(asyncio.run(manager.update_by_client_id(
            principal.client_id, {"name": "updated"}, setter
        )))
        self.assertTrue(asyncio.run(manager.delete_by_client_id(principal.client_id, setter)))
        self.assertIsNone(asyncio.run(manager.authenticate("Bearer contributor-a")))


if __name__ == "__main__":
    unittest.main()
