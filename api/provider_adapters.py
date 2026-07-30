import json
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional

import httpx

try:
    from .agnes_provider import (
        execute_agnes_chat_request,
        execute_agnes_request,
        query_agnes_video_result,
        sanitize_agnes_response,
    )
    from .model_catalog import AGNES_VIDEO_MODEL
    from .groq_provider import (
        execute_groq_request,
        groq_pool,
        sanitize_groq_response,
    )
except ImportError:
    from agnes_provider import (
        execute_agnes_chat_request,
        execute_agnes_request,
        query_agnes_video_result,
        sanitize_agnes_response,
    )
    from model_catalog import AGNES_VIDEO_MODEL
    from groq_provider import (
        execute_groq_request,
        groq_pool,
        sanitize_groq_response,
    )


JsonNormalizer = Callable[[Any, str], Any]
SuccessRecorder = Callable[[int], Awaitable[None]]
STANDARD_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "function_call"}


class ManagedStream:
    def __init__(self, response: httpx.Response, model: str, request_id: str = ""):
        self.response = response
        self.model = model
        if request_id:
            self.request_id = request_id if request_id.startswith("chatcmpl-") else f"chatcmpl-{request_id}"
        else:
            self.request_id = f"chatcmpl-{int(time.time() * 1000)}"
        self.created = int(time.time())
        self.usage: Optional[Dict[str, Any]] = None
        self.generated_text = ""
        self.sample_chunks = []

    def __aiter__(self) -> AsyncGenerator[bytes, None]:
        return self._iterate()

    async def _iterate(self) -> AsyncGenerator[bytes, None]:
        async for line in self.response.aiter_lines():
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
                if not isinstance(chunk, dict):
                    raise ValueError("SSE payload is not an object")
                chunk.setdefault("id", self.request_id)
                chunk.setdefault("object", "chat.completion.chunk")
                chunk.setdefault("created", self.created)
                chunk["model"] = self.model
                if chunk.get("usage"):
                    self.usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    reason = choice.get("finish_reason")
                    if reason is not None and reason not in STANDARD_FINISH_REASONS:
                        choice["finish_reason"] = "stop"
                    content = choice.get("delta", {}).get("content")
                    if content:
                        self.generated_text += content
                output = "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                if len(self.sample_chunks) < 10:
                    self.sample_chunks.append(output)
                yield output.encode("utf-8")
            except Exception:
                yield (line + "\n\n").encode("utf-8")


@dataclass
class ProviderResult:
    response: Optional[httpx.Response]
    provider: str
    model: str
    credential_suffix: str = ""
    site: str = ""
    route_id: Optional[str] = None
    error: Optional[str] = None
    _json_normalizer: Optional[JsonNormalizer] = None
    _success_recorder: Optional[SuccessRecorder] = None

    @property
    def available(self) -> bool:
        return self.response is not None

    @property
    def status_code(self) -> Optional[int]:
        return self.response.status_code if self.response is not None else None

    def json(self, request_id: str = "") -> Any:
        if self.response is None:
            raise RuntimeError(self.error or f"{self.provider} returned no response")
        data = self.response.json()
        return self._json_normalizer(data, request_id) if self._json_normalizer else data

    def stream(self, request_id: str = "") -> ManagedStream:
        if self.response is None:
            raise RuntimeError(self.error or f"{self.provider} returned no response")
        return ManagedStream(self.response, self.model, request_id)

    async def record_success(self, total_tokens: int = 0) -> None:
        if self._success_recorder is not None:
            await self._success_recorder(total_tokens)

    async def close(self) -> None:
        if self.response is not None:
            await self.response.aclose()


class GroqAdapter:
    def __init__(
        self,
        executor=execute_groq_request,
        pool=groq_pool,
        redis_url: str = "",
        redis_token: str = "",
    ):
        self._executor = executor
        self._pool = pool
        self._redis_url = redis_url
        self._redis_token = redis_token

    async def chat(
        self,
        client: httpx.AsyncClient,
        body: Dict[str, Any],
        show_thinking: bool = False,
    ) -> ProviderResult:
        response, model, key = await self._executor(client, body, show_thinking)
        resolved_model = model or str(body.get("model", ""))

        async def record_success(total_tokens: int) -> None:
            if not key or not model:
                return
            await self._pool.record_success(key, model, total_tokens)
            await self._pool.save_to_upstash(
                client, self._redis_url, self._redis_token
            )

        return ProviderResult(
            response=response,
            provider="groq",
            model=resolved_model,
            credential_suffix=key[-4:] if key else "",
            error=None if response is not None else "All Groq keys/models failed",
            _json_normalizer=lambda data, request_id: sanitize_groq_response(data),
            _success_recorder=record_success,
        )


class AgnesAdapter:
    def __init__(
        self,
        chat_executor=execute_agnes_chat_request,
        request_executor=execute_agnes_request,
        video_query_executor=query_agnes_video_result,
    ):
        self._chat_executor = chat_executor
        self._request_executor = request_executor
        self._video_query_executor = video_query_executor

    @staticmethod
    def _result(
        response: Optional[httpx.Response],
        candidate: Any,
        error: Optional[str],
        public_model: str,
        *,
        chat: bool,
    ) -> ProviderResult:
        def normalize_media(data: Any) -> Any:
            if isinstance(data, dict) and "model" in data:
                data = dict(data)
                data["model"] = public_model
            return data

        return ProviderResult(
            response=response,
            provider="agnes",
            model=public_model,
            credential_suffix=candidate.key[-4:] if candidate else "",
            site=candidate.site if candidate else "",
            route_id=candidate.affinity_id if candidate else None,
            error=error,
            _json_normalizer=(
                (lambda data, request_id: sanitize_agnes_response(
                    data, public_model, request_id
                ))
                if chat
                else lambda data, request_id: normalize_media(data)
            ),
        )

    async def chat(
        self,
        client: httpx.AsyncClient,
        body: Dict[str, Any],
        timeout: Any = 60.0,
    ) -> ProviderResult:
        response, candidate, error = await self._chat_executor(
            client, body, timeout=timeout
        )
        return self._result(
            response, candidate, error, str(body.get("model", "")), chat=True
        )

    async def request(
        self,
        client: httpx.AsyncClient,
        method: str,
        endpoint_path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
        public_model: str,
        stream: bool = False,
        timeout: Any = 60.0,
    ) -> ProviderResult:
        response, candidate, error = await self._request_executor(
            client,
            method,
            endpoint_path,
            body=body,
            query=query,
            public_model=public_model,
            stream=stream,
            timeout=timeout,
        )
        return self._result(
            response, candidate, error, public_model, chat=False
        )

    async def query_video(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        *,
        query: Optional[Dict[str, Any]] = None,
        timeout: Any = 60.0,
        affinity_id: Optional[str] = None,
        candidate_index: Optional[int] = None,
    ) -> ProviderResult:
        response, candidate, error = await self._video_query_executor(
            client,
            task_id,
            query=query,
            timeout=timeout,
            affinity_id=affinity_id,
            candidate_index=candidate_index,
        )
        return self._result(
            response, candidate, error, AGNES_VIDEO_MODEL, chat=False
        )
