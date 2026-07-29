import asyncio
import copy
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional, Set, Tuple

import httpx


AGNES_TEXT_MODEL = "agnes/agnes-2.5-flash"
AGNES_IMAGE_MODEL = "agnes/agnes-image-2.1-flash"
AGNES_VIDEO_MODEL = "agnes/agnes-video-v2.0"
AGNES_MODELS = [AGNES_TEXT_MODEL, AGNES_IMAGE_MODEL, AGNES_VIDEO_MODEL]

AGNES_CN_BASE_URL = "https://api.agnes-ai.cn/v1"
AGNES_INTL_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_COOLDOWN_SECONDS = 60.0

STANDARD_FINISH_REASONS = {
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
}


def _env_keys(name: str) -> List[str]:
    return [key.strip() for key in os.getenv(name, "").split(",") if key.strip()]


AGNES_CN_API_KEYS = _env_keys("AGNES_CN_API_KEYS")
AGNES_INTL_API_KEYS = _env_keys("AGNES_INTL_API_KEYS")


@dataclass(frozen=True)
class AgnesCandidate:
    site: str
    base_url: str
    key: str
    index: int

    @property
    def root_url(self) -> str:
        return self.base_url.removesuffix("/v1")


def _rate_bucket(public_model: str, body: Optional[Dict[str, Any]]) -> Tuple[str, int]:
    if public_model == AGNES_IMAGE_MODEL:
        raw_size = str((body or {}).get("size", "1K")).upper().strip()
        size = raw_size if raw_size in {"1K", "2K", "3K", "4K"} else "1K"
        return f"{public_model}:{size}", {"1K": 20, "2K": 10, "3K": 1, "4K": 1}[size]
    if public_model == AGNES_VIDEO_MODEL:
        return public_model, 1
    return public_model, 20


class AsyncAgnesCandidatePool:
    def __init__(self, site_keys: List[Tuple[str, str, List[str]]]):
        self.candidates: List[AgnesCandidate] = []
        max_keys = max((len(keys) for _, _, keys in site_keys), default=0)
        for key_index in range(max_keys):
            for site, base_url, keys in site_keys:
                if key_index >= len(keys):
                    continue
                key = keys[key_index]
                self.candidates.append(
                    AgnesCandidate(site, base_url.rstrip("/"), key, len(self.candidates))
                )
        self.index = 0
        self._lock = asyncio.Lock()
        self._site_buckets: Dict[str, Dict[str, Deque[float]]] = {}
        for candidate in self.candidates:
            self._site_buckets.setdefault(candidate.site, {
                AGNES_TEXT_MODEL: deque(),
                f"{AGNES_IMAGE_MODEL}:1K": deque(),
                f"{AGNES_IMAGE_MODEL}:2K": deque(),
                f"{AGNES_IMAGE_MODEL}:3K": deque(),
                f"{AGNES_IMAGE_MODEL}:4K": deque(),
                AGNES_VIDEO_MODEL: deque(),
            })
        self._state: Dict[int, Dict[str, Any]] = {
            candidate.index: {
                "cooldown": 0.0,
                "request_total": 0,
            }
            for candidate in self.candidates
        }

    @property
    def key_count(self) -> int:
        return len(self.candidates)

    @property
    def site_count(self) -> int:
        return len({candidate.site for candidate in self.candidates})

    def get_counts(self) -> Dict[str, Any]:
        by_site = {"cn": 0, "intl": 0}
        for candidate in self.candidates:
            by_site[candidate.site] = by_site.get(candidate.site, 0) + 1
        return {
            "key_count": self.key_count,
            "site_count": self.site_count,
            "keys_by_site": by_site,
        }

    @staticmethod
    def _clean_timestamps(timestamps: Deque[float], now: float) -> None:
        while timestamps and now - timestamps[0] >= 60.0:
            timestamps.popleft()

    async def acquire(
        self,
        public_model: str,
        body: Optional[Dict[str, Any]] = None,
        excluded: Optional[Set[int]] = None,
    ) -> Optional[AgnesCandidate]:
        excluded = excluded or set()
        bucket_name, limit_rpm = _rate_bucket(public_model, body)
        async with self._lock:
            if not self.candidates:
                return None
            now = time.time()
            for _ in range(len(self.candidates)):
                candidate = self.candidates[self.index]
                self.index = (self.index + 1) % len(self.candidates)
                if candidate.index in excluded:
                    continue
                state = self._state[candidate.index]
                bucket = self._site_buckets[candidate.site].setdefault(bucket_name, deque())
                self._clean_timestamps(bucket, now)
                if state["cooldown"] <= now and len(bucket) < limit_rpm:
                    bucket.append(now)
                    state["request_total"] += 1
                    return candidate
        return None

    async def cooldown(self, candidate: AgnesCandidate, seconds: float = AGNES_COOLDOWN_SECONDS) -> None:
        async with self._lock:
            state = self._state.get(candidate.index)
            if state is not None:
                state["cooldown"] = max(state["cooldown"], time.time() + max(0.0, seconds))

    async def get_metrics(self) -> Dict[str, Any]:
        async with self._lock:
            now = time.time()
            candidates = []
            for candidate in self.candidates:
                state = self._state[candidate.index]
                buckets = {}
                for bucket_name, timestamps in self._site_buckets[candidate.site].items():
                    self._clean_timestamps(timestamps, now)
                    model, _, size = bucket_name.partition(":")
                    _, limit_rpm = _rate_bucket(model, {"size": size} if size else None)
                    buckets[bucket_name] = {
                        "current_rpm": len(timestamps),
                        "limit_rpm": limit_rpm,
                    }
                candidates.append(
                    {
                        "site": candidate.site,
                        "key_suffix": candidate.key[-4:],
                        "cooldown": max(0, int(state["cooldown"] - now)),
                        "request_total": state["request_total"],
                        "models": buckets,
                    }
                )
            return {**self.get_counts(), "candidates": candidates}


agnes_pool = AsyncAgnesCandidatePool(
    [
        ("cn", AGNES_CN_BASE_URL, AGNES_CN_API_KEYS),
        ("intl", AGNES_INTL_BASE_URL, AGNES_INTL_API_KEYS),
    ]
)


def _candidate_headers(candidate: AgnesCandidate) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {candidate.key}",
        "Content-Type": "application/json",
    }


def _retry_after(response: httpx.Response) -> float:
    value = response.headers.get("retry-after", "")
    try:
        return max(AGNES_COOLDOWN_SECONDS, float(value))
    except (TypeError, ValueError):
        return AGNES_COOLDOWN_SECONDS


def _upstream_body(body: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if body is None:
        return None
    result = copy.deepcopy(body)
    model = result.get("model")
    if isinstance(model, str) and model.startswith("agnes/"):
        result["model"] = model[len("agnes/") :]
    return result


async def execute_agnes_request(
    client: httpx.AsyncClient,
    method: str,
    endpoint_path: str,
    body: Optional[Dict[str, Any]] = None,
    query: Optional[Dict[str, Any]] = None,
    public_model: str = AGNES_TEXT_MODEL,
    stream: bool = False,
    timeout: Any = 60.0,
) -> Tuple[Optional[httpx.Response], Optional[AgnesCandidate], Optional[str]]:
    if public_model not in AGNES_MODELS:
        return None, None, f"Unsupported Agnes model: {public_model}"
    if not agnes_pool.candidates:
        return None, None, "No Agnes API keys configured"

    tried: Set[int] = set()
    last_error: Optional[str] = None
    last_candidate: Optional[AgnesCandidate] = None
    request_body = _upstream_body(body)
    while len(tried) < len(agnes_pool.candidates):
        candidate = await agnes_pool.acquire(public_model, body, tried)
        if candidate is None:
            break
        last_candidate = candidate
        tried.add(candidate.index)
        try:
            request = client.build_request(
                method.upper(),
                f"{candidate.base_url}/{endpoint_path.lstrip('/')}",
                json=request_body,
                params=query,
                headers=_candidate_headers(candidate),
                timeout=timeout,
            )
            response = await client.send(request, stream=stream)
        except httpx.HTTPError as exc:
            last_error = f"{candidate.site} network error: {exc}"
            continue

        if 200 <= response.status_code < 300:
            return response, candidate, None
        if response.status_code in {400, 422}:
            return response, candidate, None
        if response.status_code in {401, 403, 429}:
            await agnes_pool.cooldown(candidate, _retry_after(response))
            last_error = f"{candidate.site} upstream returned {response.status_code}"
            await response.aclose()
            continue
        if response.status_code >= 500:
            last_error = f"{candidate.site} upstream returned {response.status_code}"
            await response.aclose()
            continue

        # Other client errors are deterministic, so expose them instead of spending more keys.
        return response, candidate, None

    return None, last_candidate, last_error or "No Agnes candidate is currently available"


async def execute_agnes_chat_request(
    client: httpx.AsyncClient,
    body: Dict[str, Any],
    stream: Optional[bool] = None,
    timeout: Any = 60.0,
) -> Tuple[Optional[httpx.Response], Optional[AgnesCandidate], Optional[str]]:
    use_stream = bool(body.get("stream", False)) if stream is None else stream
    chat_body = copy.deepcopy(body)
    chat_body.setdefault("model", AGNES_TEXT_MODEL)
    chat_body["stream"] = use_stream
    return await execute_agnes_request(
        client,
        "POST",
        "/chat/completions",
        body=chat_body,
        public_model=str(chat_body["model"]),
        stream=use_stream,
        timeout=timeout,
    )


def _response_model(upstream_model: Any, public_model: str) -> str:
    if public_model:
        return public_model if public_model.startswith("agnes/") else f"agnes/{public_model}"
    model = str(upstream_model or AGNES_TEXT_MODEL)
    return model if model.startswith("agnes/") else f"agnes/{model}"


def _sanitize_choices(data: Dict[str, Any]) -> None:
    choices = data.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        reason = choice.get("finish_reason")
        if reason is not None and reason not in STANDARD_FINISH_REASONS:
            choice["finish_reason"] = "stop"


async def sanitize_agnes_sse_stream(
    response: httpx.Response,
    public_model: str = AGNES_TEXT_MODEL,
    request_id: str = "",
) -> AsyncGenerator[bytes, None]:
    completion_id = request_id or f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())
    async for line in response.aiter_lines():
        if not line:
            continue
        if not line.startswith("data:"):
            yield (line + "\n").encode("utf-8")
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            yield b"data: [DONE]\n\n"
            continue
        try:
            chunk = json.loads(payload)
            if isinstance(chunk, dict):
                chunk.setdefault("id", completion_id)
                chunk.setdefault("object", "chat.completion.chunk")
                chunk.setdefault("created", created)
                chunk["model"] = _response_model(chunk.get("model"), public_model)
                _sanitize_choices(chunk)
                yield ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8")
                continue
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        yield (line + "\n\n").encode("utf-8")


def sanitize_agnes_response(
    data: Dict[str, Any],
    public_model: str = AGNES_TEXT_MODEL,
    request_id: str = "",
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return data
    result = copy.deepcopy(data)
    result.setdefault("id", request_id or f"chatcmpl-{int(time.time() * 1000)}")
    result.setdefault("object", "chat.completion")
    result.setdefault("created", int(time.time()))
    result["model"] = _response_model(result.get("model"), public_model)
    _sanitize_choices(result)
    return result


async def query_agnes_video_result(
    client: httpx.AsyncClient,
    task_id: str,
    query: Optional[Dict[str, Any]] = None,
    timeout: Any = 60.0,
    allow_legacy: bool = True,
    candidate_index: Optional[int] = None,
) -> Tuple[Optional[httpx.Response], Optional[AgnesCandidate], Optional[str]]:
    if not agnes_pool.candidates:
        return None, None, "No Agnes API keys configured"
    params = dict(query or {})
    params.setdefault("video_id", task_id)
    candidates = [
        candidate for candidate in agnes_pool.candidates
        if candidate_index is None or candidate.index == candidate_index
    ]
    routes = [(candidate, f"{candidate.root_url}/agnesapi", params) for candidate in candidates]
    if allow_legacy:
        routes.extend(
            (candidate, f"{candidate.base_url}/videos/{task_id}", query)
            for candidate in candidates
        )

    last_error: Optional[str] = None
    last_candidate: Optional[AgnesCandidate] = None
    for candidate, url, route_query in routes:
        last_candidate = candidate
        try:
            request = client.build_request(
                "GET",
                url,
                params=route_query,
                headers=_candidate_headers(candidate),
                timeout=timeout,
            )
            response = await client.send(request)
        except httpx.HTTPError as exc:
            last_error = f"{candidate.site} network error: {exc}"
            continue
        if 200 <= response.status_code < 300 or response.status_code in {400, 422}:
            return response, candidate, None
        if response.status_code in {401, 403, 429}:
            await agnes_pool.cooldown(candidate, _retry_after(response))
        last_error = f"{candidate.site} upstream returned {response.status_code}"
        await response.aclose()

    return None, last_candidate, last_error or "No Agnes video result endpoint is available"


async def get_agnes_metrics() -> Dict[str, Any]:
    return await agnes_pool.get_metrics()


def get_agnes_counts() -> Dict[str, Any]:
    return agnes_pool.get_counts()
