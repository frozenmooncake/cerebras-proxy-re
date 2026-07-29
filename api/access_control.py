import asyncio
import hashlib
import inspect
import json
import os
import secrets
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Deque, Dict, List, Mapping, Optional, Set, Tuple

import httpx


POLICIES_STORAGE_KEY = "gateway_access_policies"
AGNES_IMAGE_MODEL = "agnes/agnes-image-2.1-flash"
AGNES_VIDEO_MODEL = "agnes/agnes-video-v2.0"
IMAGE_LIMITS = {"1K": 20, "2K": 10, "3K": 1, "4K": 1}
CEREBRAS_MODELS = {"gemma-4-31b", "zai-glm-4.7", "gpt-oss-120b"}


@dataclass(frozen=True)
class ClientPrincipal:
    client_id: str
    key_suffix: str
    name: str
    providers: Dict[str, int]
    allowed_models: Optional[Set[str]]
    enabled: bool
    is_open: bool = False


def _client_id(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]


def _suffix(secret: str) -> str:
    return secret[-6:] if len(secret) > 6 else secret


def _stored_client_id(identifier: str) -> str:
    if identifier.startswith("sha256:"):
        return identifier[7:23]
    return _client_id(identifier)


def _normalize_policy(raw: Any) -> Dict[str, Any]:
    value = raw if isinstance(raw, Mapping) else {}
    providers = {}
    raw_providers = value.get("providers")
    if isinstance(raw_providers, Mapping):
        for provider in ("cerebras", "groq", "agnes"):
            try:
                providers[provider] = max(0, min(100, int(raw_providers.get(provider, 0))))
            except (TypeError, ValueError):
                providers[provider] = 0
    else:
        try:
            legacy_contributions = max(1, min(100, int(value.get("contributions", 1))))
        except (TypeError, ValueError):
            legacy_contributions = 1
        legacy_scope = str(value.get("scope", "all")).strip().lower()
        providers = {
            "cerebras": legacy_contributions if legacy_scope == "all" else 0,
            "groq": legacy_contributions if legacy_scope == "all" else 0,
            "agnes": legacy_contributions,
        }
    raw_models = value.get("allowed_models")
    allowed_models = None
    if isinstance(raw_models, (list, tuple, set)):
        models = sorted(set(str(model).strip() for model in raw_models if str(model).strip()))
        allowed_models = models or None
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        enabled = str(enabled).strip().lower() not in ("0", "false", "no", "off")
    return {
        "name": str(value.get("name", "")).strip(),
        "providers": providers,
        "allowed_models": allowed_models,
        "enabled": enabled,
        "_key_suffix": str(value.get("_key_suffix", "")).strip(),
    }


def _parse_structured(value: str) -> Dict[str, Dict[str, Any]]:
    if not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    result = {}
    if isinstance(parsed, Mapping):
        items = parsed.items()
    elif isinstance(parsed, list):
        items = ((item.get("key"), item) for item in parsed if isinstance(item, Mapping))
    else:
        return {}
    for secret, policy in items:
        if isinstance(secret, str) and secret:
            result[secret] = _normalize_policy(policy)
    return result


async def _call(callback: Callable[..., Any], *args: Any) -> Any:
    result = callback(*args)
    if inspect.isawaitable(result):
        return await result
    return result


class AccessManager:
    def __init__(self) -> None:
        legacy = {
            key.strip(): _normalize_policy({"providers": {"cerebras": 1, "groq": 1, "agnes": 1}})
            for key in os.getenv("CUSTOM_API_KEYS", "").split(",")
            if key.strip()
        }
        legacy.update(_parse_structured(os.getenv("GATEWAY_KEYS_JSON", "")))
        self._environment_policies: Dict[str, Dict[str, Any]] = legacy
        self._persisted_policies: Dict[str, Dict[str, Any]] = {}
        self._policies: Dict[str, Dict[str, Any]] = dict(legacy)
        self._lock = asyncio.Lock()
        self._quota_lock = asyncio.Lock()
        self._quota: Dict[Tuple[str, str], Deque[float]] = {}
        self._token_quota: Dict[Tuple[str, str], Deque[Tuple[float, int]]] = {}
        self._storage_unavailable = False
        self._last_refresh = 0.0

    @staticmethod
    def provider_for_model(model: str) -> str:
        if model.startswith("agnes/"):
            return "agnes"
        if model in CEREBRAS_MODELS:
            return "cerebras"
        return "groq"

    @staticmethod
    def _principal(secret: str, policy: Mapping[str, Any]) -> ClientPrincipal:
        models = policy.get("allowed_models")
        return ClientPrincipal(
            client_id=_client_id(secret),
            key_suffix=_suffix(secret),
            name=str(policy.get("name", "")),
            providers=dict(policy.get("providers", {})),
            allowed_models=set(models) if models else None,
            enabled=bool(policy.get("enabled", True)),
            is_open=False,
        )

    @staticmethod
    def _open_principal() -> ClientPrincipal:
        return ClientPrincipal("open", "", "Open access", {"cerebras": 1, "groq": 1, "agnes": 1}, None, True, True)

    async def initialize(self, getter: Callable[[str], Any]) -> None:
        try:
            stored = await _call(getter, POLICIES_STORAGE_KEY)
        except Exception:
            async with self._lock:
                self._storage_unavailable = True
                self._persisted_policies = {}
                self._policies = dict(self._environment_policies)
            return
        if isinstance(stored, str):
            try:
                stored = json.loads(stored)
            except (TypeError, ValueError):
                stored = None
        parsed = self._parse_persisted(stored)
        async with self._lock:
            self._persisted_policies = parsed
            self._policies = dict(self._environment_policies)
            self._policies.update(parsed)
            self._storage_unavailable = False
            self._last_refresh = time.monotonic()

    async def refresh_if_due(self, getter: Callable[[str], Any], interval: float = 5.0) -> None:
        async with self._lock:
            due = time.monotonic() - self._last_refresh >= interval
            if due:
                self._last_refresh = time.monotonic()
        if due:
            await self.initialize(getter)

    @staticmethod
    def _parse_persisted(stored: Any) -> Dict[str, Dict[str, Any]]:
        if isinstance(stored, Mapping):
            return {
                secret: _normalize_policy(policy)
                for secret, policy in stored.items()
                if isinstance(secret, str) and secret
            }
        if isinstance(stored, list):
            return {
                item["key"]: _normalize_policy(item)
                for item in stored
                if isinstance(item, Mapping) and isinstance(item.get("key"), str) and item["key"]
            }
        return {}

    async def auth_enabled(self) -> bool:
        async with self._lock:
            return bool(self._policies)

    async def authenticate(self, authorization: Optional[str]) -> Optional[ClientPrincipal]:
        async with self._lock:
            if self._storage_unavailable:
                return None
            if not self._policies:
                return self._open_principal()
            if not isinstance(authorization, str):
                return None
            scheme, separator, secret = authorization.strip().partition(" ")
            if not separator or scheme.lower() != "bearer" or not secret.strip():
                return None
            secret = secret.strip()
            policy = self._policies.get(secret)
            if policy is None:
                digest_key = "sha256:" + hashlib.sha256(secret.encode("utf-8")).hexdigest()
                policy = self._policies.get(digest_key)
            if policy is None or not policy.get("enabled", True):
                return None
            return self._principal(secret, policy)

    def authorize(self, principal: Optional[ClientPrincipal], model: str, provider: Optional[str] = None) -> bool:
        if principal is None or not principal.enabled:
            return False
        if principal.is_open:
            return True
        if not provider:
            provider = self.provider_for_model(model)
        if not provider or principal.providers.get(provider, 0) <= 0:
            return False
        return principal.allowed_models is None or model in principal.allowed_models

    async def export(self, merged: bool = False) -> Dict[str, Dict[str, Any]]:
        async with self._lock:
            source = self._policies if merged else self._persisted_policies
            return {secret: dict(policy) for secret, policy in source.items()}

    async def _persist(self, setter: Callable[..., Any], policies: Dict[str, Dict[str, Any]]) -> bool:
        try:
            result = await _call(setter, POLICIES_STORAGE_KEY, policies)
            return result is not False
        except Exception:
            return False

    async def upsert(
        self, secret: str, policy: Mapping[str, Any], setter: Callable[..., Any]
    ) -> bool:
        if not isinstance(secret, str) or not secret:
            return False
        normalized = _normalize_policy(policy)
        async with self._lock:
            updated = dict(self._persisted_policies)
            updated[secret] = normalized
            if not await self._persist(setter, updated):
                return False
            self._persisted_policies = updated
            self._policies = dict(self._environment_policies)
            self._policies.update(updated)
            return True

    async def delete(self, secret: str, setter: Callable[..., Any]) -> bool:
        async with self._lock:
            if secret not in self._policies:
                return False
            updated = dict(self._persisted_policies)
            if secret in self._environment_policies:
                disabled = dict(self._policies[secret])
                disabled["enabled"] = False
                updated[secret] = disabled
            else:
                updated.pop(secret, None)
            if not await self._persist(setter, updated):
                return False
            self._persisted_policies = updated
            self._policies = dict(self._environment_policies)
            self._policies.update(updated)
            return True

    async def create(
        self, policy: Mapping[str, Any], setter: Callable[..., Any]
    ) -> Optional[Dict[str, Any]]:
        normalized = _normalize_policy(policy)
        secret = "cpr_multi_c{0}_{1}".format(
            sum(normalized["providers"].values()), secrets.token_urlsafe(24)
        )
        normalized["_key_suffix"] = _suffix(secret)
        identifier = "sha256:" + hashlib.sha256(secret.encode("utf-8")).hexdigest()
        if not await self.upsert(identifier, normalized, setter):
            return None
        return {"key": secret, "policy": dict(normalized)}

    async def update(
        self, secret: str, policy: Mapping[str, Any], setter: Callable[..., Any]
    ) -> bool:
        async with self._lock:
            current = self._policies.get(secret)
            if current is None:
                return False
            merged = dict(current)
            merged.update(policy)
            updated = dict(self._persisted_policies)
            updated[secret] = _normalize_policy(merged)
            if not await self._persist(setter, updated):
                return False
            self._persisted_policies = updated
            self._policies = dict(self._environment_policies)
            self._policies.update(updated)
            return True

    async def list_clients(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [
                {
                    "client_id": _stored_client_id(secret),
                    "key_suffix": policy.get("_key_suffix") or _suffix(secret),
                    "name": policy["name"],
                    "providers": dict(policy["providers"]),
                    "allowed_models": list(policy["allowed_models"]) if policy["allowed_models"] else None,
                    "enabled": policy["enabled"],
                }
                for secret, policy in self._policies.items()
            ]

    def _secret_by_client_id(self, client_id: str) -> Optional[str]:
        for secret in self._policies:
            if _stored_client_id(secret) == client_id:
                return secret
        return None

    async def update_by_client_id(
        self, client_id: str, policy: Mapping[str, Any], setter: Callable[..., Any]
    ) -> bool:
        async with self._lock:
            secret = self._secret_by_client_id(client_id)
            if secret is None:
                return False
            current = dict(self._policies[secret])
            current.update(policy)
            updated = dict(self._persisted_policies)
            updated[secret] = _normalize_policy(current)
            if not await self._persist(setter, updated):
                return False
            self._persisted_policies = updated
            self._policies = dict(self._environment_policies)
            self._policies.update(updated)
            return True

    async def delete_by_client_id(self, client_id: str, setter: Callable[..., Any]) -> bool:
        async with self._lock:
            secret = self._secret_by_client_id(client_id)
            if secret is None:
                return False
            updated = dict(self._persisted_policies)
            if secret in self._environment_policies:
                disabled = dict(self._policies[secret])
                disabled["enabled"] = False
                updated[secret] = disabled
            else:
                updated.pop(secret, None)
            if not await self._persist(setter, updated):
                return False
            self._persisted_policies = updated
            self._policies = dict(self._environment_policies)
            self._policies.update(updated)
            return True

    @staticmethod
    def _bucket(model: str, body: Optional[Mapping[str, Any]]) -> Tuple[str, str, int, Optional[int]]:
        if model == AGNES_IMAGE_MODEL:
            size = str((body or {}).get("size", "1K")).strip().upper()
            if size not in IMAGE_LIMITS:
                size = "1K"
            return "agnes", model + ":" + size, IMAGE_LIMITS[size], None
        if model == AGNES_VIDEO_MODEL:
            return "agnes", model, 1, None
        provider = AccessManager.provider_for_model(model)
        if provider == "agnes":
            return provider, model, 20, None
        if provider == "cerebras":
            return provider, provider, 5, 30000
        return provider, provider, 30, 40000

    @staticmethod
    def _redis_result(data: Any, index: int) -> Optional[int]:
        if isinstance(data, Mapping):
            data = data.get("result")
        if not isinstance(data, list) or not data:
            return None
        if index >= len(data):
            return None
        first = data[index]
        if isinstance(first, Mapping):
            first = first.get("result")
        try:
            return int(first)
        except (TypeError, ValueError):
            return None

    async def _consume_redis(
        self,
        principal: ClientPrincipal,
        bucket: str,
        client: httpx.AsyncClient,
        redis_url: str,
        redis_token: str,
        estimated_tokens: int,
    ) -> Optional[Tuple[int, int]]:
        minute = int(time.time() // 60)
        digest = hashlib.sha256((principal.client_id + "\0" + bucket).encode("utf-8")).hexdigest()
        key = "cpr:quota:{0}:{1}".format(digest, minute)
        try:
            response = await client.post(
                redis_url.rstrip("/") + "/pipeline",
                json=[
                    ["INCR", key + ":requests"],
                    ["EXPIRE", key + ":requests", 60, "NX"],
                    ["INCRBY", key + ":tokens", max(0, estimated_tokens)],
                    ["EXPIRE", key + ":tokens", 60, "NX"],
                ],
                headers={"Authorization": "Bearer " + redis_token},
                timeout=3.0,
            )
            response.raise_for_status()
            data = response.json()
            requests = self._redis_result(data, 0)
            tokens = self._redis_result(data, 2)
            return (requests, tokens) if requests is not None and tokens is not None else None
        except Exception:
            return None

    async def consume(
        self,
        principal: ClientPrincipal,
        model: str,
        body: Optional[Mapping[str, Any]] = None,
        client: Optional[httpx.AsyncClient] = None,
        redis_url: Optional[str] = None,
        redis_token: Optional[str] = None,
        estimated_tokens: int = 0,
    ) -> Dict[str, Any]:
        if principal.is_open:
            return {"allowed": True, "current_rpm": 0, "limit_rpm": None, "current_tpm": 0, "limit_tpm": None, "retry_after": 0}
        provider, bucket, official_rpm, official_tpm = self._bucket(model, body)
        contributions = principal.providers.get(provider, 0)
        if contributions <= 0:
            return {"allowed": False, "current_rpm": 0, "limit_rpm": 0, "current_tpm": 0, "limit_tpm": 0, "retry_after": 60}
        limit = official_rpm * contributions * 2
        token_limit = official_tpm * contributions * 2 if official_tpm else None
        if client is not None and redis_url and redis_token:
            counts = await self._consume_redis(principal, bucket, client, redis_url, redis_token, estimated_tokens)
            if counts is not None:
                count, token_count = counts
                allowed = count <= limit and (token_limit is None or token_count <= token_limit)
                return {
                    "allowed": allowed,
                    "current_rpm": count,
                    "limit_rpm": limit,
                    "current_tpm": token_count,
                    "limit_tpm": token_limit,
                    "retry_after": 0 if allowed else 60,
                }
        now = time.monotonic()
        async with self._quota_lock:
            queue = self._quota.setdefault((principal.client_id, bucket), deque())
            token_queue = self._token_quota.setdefault((principal.client_id, bucket), deque())
            cutoff = now - 60.0
            while queue and queue[0] <= cutoff:
                queue.popleft()
            while token_queue and token_queue[0][0] <= cutoff:
                token_queue.popleft()
            current_tokens = sum(item[1] for item in token_queue)
            if len(queue) < limit and (token_limit is None or current_tokens + estimated_tokens <= token_limit):
                queue.append(now)
                if estimated_tokens > 0:
                    token_queue.append((now, estimated_tokens))
                allowed = True
            else:
                allowed = False
            retry_after = 0 if allowed else (max(1, int(61.0 - (now - queue[0]))) if queue else 60)
            return {
                "allowed": allowed,
                "current_rpm": len(queue),
                "limit_rpm": limit,
                "current_tpm": current_tokens + (estimated_tokens if allowed else 0),
                "limit_tpm": token_limit,
                "retry_after": retry_after,
            }

    async def record_tokens(
        self,
        principal: ClientPrincipal,
        model: str,
        tokens: int,
        body: Optional[Mapping[str, Any]] = None,
        client: Optional[httpx.AsyncClient] = None,
        redis_url: Optional[str] = None,
        redis_token: Optional[str] = None,
    ) -> None:
        if principal.is_open or tokens <= 0:
            return
        _, bucket, _, token_limit = self._bucket(model, body)
        if token_limit is None:
            return
        if client is not None and redis_url and redis_token:
            minute = int(time.time() // 60)
            digest = hashlib.sha256((principal.client_id + "\0" + bucket).encode("utf-8")).hexdigest()
            key = "cpr:quota:{0}:{1}:tokens".format(digest, minute)
            try:
                response = await client.post(
                    redis_url.rstrip("/") + "/pipeline",
                    json=[["INCRBY", key, tokens], ["EXPIRE", key, 60, "NX"]],
                    headers={"Authorization": "Bearer " + redis_token}, timeout=3.0,
                )
                response.raise_for_status()
                return
            except Exception:
                pass
        async with self._quota_lock:
            queue = self._token_quota.setdefault((principal.client_id, bucket), deque())
            queue.append((time.monotonic(), tokens))

    async def quota_snapshots(self) -> List[Dict[str, Any]]:
        now = time.monotonic()
        snapshots = []
        async with self._quota_lock:
            for (client_id, bucket), queue in self._quota.items():
                cutoff = now - 60.0
                while queue and queue[0] <= cutoff:
                    queue.popleft()
                if queue:
                    snapshots.append(
                        {"client_id": client_id, "bucket": bucket, "current_rpm": len(queue)}
                    )
        return snapshots

    async def get_quota_snapshots(self) -> List[Dict[str, Any]]:
        return await self.quota_snapshots()


access_manager = AccessManager()
