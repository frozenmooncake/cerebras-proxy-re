import os
import json
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional, Any, List, AsyncGenerator
import httpx

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq 官方托管模型全局汇总 (已按最新列表更新)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3.6-27b"
]

GROQ_API_KEYS = list(filter(None, os.getenv("GROQ_API_KEYS", "").split(",")))
KEY_COOLDOWN = 60

STANDARD_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter"}

class AsyncGroqKeyPool:
    def __init__(self, keys: list):
        self.keys = keys
        self.index = 0
        self.data = {}
        self.last_save_time = 0

    async def restore_from_upstash(self, client: httpx.AsyncClient, redis_url: str, redis_token: str):
        history_pool = {}
        if redis_url and redis_token:
            try:
                r = await client.get(f"{redis_url}/get/groq_gateway_pool", headers={"Authorization": f"Bearer {redis_token}"}, timeout=3.0)
                if r.status_code == 200 and r.json().get("result"):
                    history_pool = json.loads(r.json()["result"])
            except Exception:
                pass

        bj_now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")

        for key in self.keys:
            self.data[key] = {}
            for model in GROQ_MODELS:
                h_req = history_pool.get(key, {}).get(model, {}).get("requests", 0)
                h_tok = history_pool.get(key, {}).get(model, {}).get("tokens", 0)
                h_tpd = history_pool.get(key, {}).get(model, {}).get("tpd_tokens", 0)
                h_date = history_pool.get(key, {}).get(model, {}).get("last_reset_date", "")

                if h_date != bj_now:
                    h_tpd = 0
                    h_date = bj_now

                self.data[key][model] = {
                    "cooldown": 0,
                    "requests": h_req,
                    "tokens": h_tok,
                    "tpd_tokens": h_tpd,
                    "last_reset_date": h_date,
                    "req_timestamps": deque(),
                    "token_timestamps": deque(),
                    "limit_rpm": 30,
                    "limit_rpd": 14400,
                    "limit_tpm": 40000,
                    "limit_tpd": 1000000,
                    "has_synced": False
                }

    async def save_to_upstash(self, client: httpx.AsyncClient, redis_url: str, redis_token: str, force: bool = False):
        now = time.time()
        if not force and (now - self.last_save_time < 15):
            return

        self.last_save_time = now
        try:
            export = {}
            for k, v in self.data.items():
                export[k] = {}
                for m, info in v.items():
                    export[k][m] = {
                        "requests": info["requests"],
                        "tokens": info["tokens"],
                        "tpd_tokens": info.get("tpd_tokens", 0),
                        "last_reset_date": info.get("last_reset_date", "")
                    }
            if redis_url and redis_token:
                await client.post(f"{redis_url}/set/groq_gateway_pool", headers={"Authorization": f"Bearer {redis_token}"}, content=json.dumps(export), timeout=3.0)
        except Exception:
            pass

    def clean_windows(self, info: dict, now: float):
        while info["req_timestamps"] and now - info["req_timestamps"][0] > 60:
            info["req_timestamps"].popleft()
        while info["token_timestamps"] and now - info["token_timestamps"][0][0] > 60:
            info["token_timestamps"].popleft()

    def get_current_metrics(self, key: str, model: str) -> dict:
        now = time.time()
        if key not in self.data or model not in self.data[key]:
            return {"current_rpm": 0, "limit_rpm": 30, "current_rpd": 0, "limit_rpd": 14400, "current_tpm": 0, "limit_tpm": 40000, "current_tpd": 0, "limit_tpd": 1000000}

        info = self.data[key][model]
        self.clean_windows(info, now)

        bj_now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
        if info.get("last_reset_date") != bj_now:
            info["tpd_tokens"] = 0
            info["last_reset_date"] = bj_now

        return {
            "current_rpm": len(info["req_timestamps"]), "limit_rpm": info["limit_rpm"],
            "current_rpd": info["requests"], "limit_rpd": info["limit_rpd"],
            "current_tpm": sum(t[1] for t in info["token_timestamps"]), "limit_tpm": info["limit_tpm"],
            "current_tpd": info.get("tpd_tokens", 0), "limit_tpd": info["limit_tpd"]
        }

    def get_next_key(self, target_model: str, exclude_keys: set = None) -> Tuple[Optional[str], Optional[str]]:
        if not self.keys:
            return None, None
        if exclude_keys is None:
            exclude_keys = set()

        now = time.time()
        num_keys = len(self.keys)
        for _ in range(num_keys):
            key = self.keys[self.index]
            self.index = (self.index + 1) % num_keys
            if key in exclude_keys:
                continue

            info = self.data.get(key, {}).get(target_model)
            if not info:
                continue

            self.clean_windows(info, now)
            if info["cooldown"] <= now and len(info["req_timestamps"]) < info["limit_rpm"]:
                return key, target_model

        return None, None

    def record_cooldown(self, key: str, model: str):
        if key in self.data and model in self.data[key]:
            self.data[key][model]["cooldown"] = time.time() + KEY_COOLDOWN

    def record_request_attempt(self, key: str, model: str, estimated_tokens: int = 0):
        now = time.time()
        if key in self.data and model in self.data[key]:
            info = self.data[key][model]
            info["req_timestamps"].append(now)
            if estimated_tokens > 0:
                info["token_timestamps"].append((now, estimated_tokens))

    def sync_headers(self, key: str, model: str, headers: httpx.Headers, actual_tokens: int = 0):
        if not key or model not in GROQ_MODELS or key not in self.data:
            return
        now = time.time()
        bj_now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")

        try:
            limit_rpd = headers.get("x-ratelimit-limit-requests") or headers.get("x-ratelimit-limit-requests-day")
            limit_tpm = headers.get("x-ratelimit-limit-tokens") or headers.get("x-ratelimit-limit-tokens-minute")
            rem_rpd = headers.get("x-ratelimit-remaining-requests") or headers.get("x-ratelimit-remaining-requests-day")
            rem_tpm = headers.get("x-ratelimit-remaining-tokens") or headers.get("x-ratelimit-remaining-tokens-minute")

            info = self.data[key][model]
            if limit_rpd is not None: info["limit_rpd"] = int(limit_rpd)
            if limit_tpm is not None: info["limit_tpm"] = int(limit_tpm)

            if rem_tpm is not None:
                used_tpm = max(0, info["limit_tpm"] - int(rem_tpm))
                info["token_timestamps"].clear()
                info["token_timestamps"].append((now, used_tpm))

            if rem_rpd is not None:
                info["requests"] = max(info["requests"], info["limit_rpd"] - int(rem_rpd))

            if actual_tokens > 0:
                if info.get("last_reset_date") != bj_now:
                    info["tpd_tokens"] = 0
                    info["last_reset_date"] = bj_now
                info["tokens"] += actual_tokens
                info["tpd_tokens"] += actual_tokens

            info["has_synced"] = True
        except Exception:
            if actual_tokens > 0:
                info = self.data[key][model]
                if info.get("last_reset_date") != bj_now:
                    info["tpd_tokens"] = 0
                    info["last_reset_date"] = bj_now
                info["tokens"] += actual_tokens
                info["tpd_tokens"] += actual_tokens

groq_pool = AsyncGroqKeyPool(GROQ_API_KEYS)

def sanitize_groq_body(raw_body: dict, show_thinking: bool, target_model: str) -> dict:
    """清理并适配 Groq 请求体"""
    body = raw_body.copy()
    body["model"] = target_model
    body["stream"] = bool(raw_body.get("stream", False))

    if body["stream"]:
        body["stream_options"] = {"include_usage": True}

    # 兼容 GPT-OSS / Qwen / Llama 针对思考过程的处理
    if not show_thinking:
        body["reasoning_format"] = "hidden"
    else:
        body["reasoning_format"] = "raw"

    for extra in ["thinkingdisplay", "thinking"]:
        body.pop(extra, None)

    return body

async def sanitize_sse_stream(response: httpx.Response) -> AsyncGenerator[bytes, None]:
    """
    关键修复：拦截 SSE 流中的 finish_reason，如果为 'other' 或非法值，替换为 'stop'
    彻底消除 Cherry Studio / Vercel AI SDK 的 AI_FinishReasonError 报错
    """
    async for line in response.aiter_lines():
        if not line:
            yield b"\n\n"
            continue

        if line.startswith("data: "):
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                yield b"data: [DONE]\n\n"
                continue

            try:
                chunk = json.loads(data_str)
                modified = False
                if "choices" in chunk and isinstance(chunk["choices"], list):
                    for choice in chunk["choices"]:
                        reason = choice.get("finish_reason")
                        if reason and reason not in STANDARD_FINISH_REASONS:
                            # 将 "other" 或非标 finish_reason 修正为 "stop"
                            choice["finish_reason"] = "stop"
                            modified = True

                if modified:
                    line = f"data: {json.dumps(chunk, ensure_ascii=False)}"
            except Exception:
                pass

        yield (line + "\n\n").encode("utf-8")

async def execute_groq_request(
    client: httpx.AsyncClient, 
    raw_body: dict, 
    show_thinking: bool = False
) -> Tuple[Optional[httpx.Response], Optional[str], Optional[str]]:
    """
    执行 Groq 请求并做兼容修饰
    """
    if not GROQ_API_KEYS:
        return None, None, None

    request_model = raw_body.get("model")
    models_to_try = [request_model] if request_model in GROQ_MODELS else GROQ_MODELS

    for target_model in models_to_try:
        tried_keys = set()
        for _ in range(len(GROQ_API_KEYS)):
            key, _ = groq_pool.get_next_key(target_model, tried_keys)
            if not key:
                break

            tried_keys.add(key)
            groq_body = sanitize_groq_body(raw_body, show_thinking, target_model)
            groq_pool.record_request_attempt(key, target_model)

            try:
                req = client.build_request(
                    "POST",
                    f"{GROQ_BASE_URL}/chat/completions",
                    json=groq_body,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"
                    },
                    timeout=60.0
                )
                response = await client.send(req, stream=groq_body.get("stream", False))

                if response.status_code == 200:
                    groq_pool.sync_headers(key, target_model, response.headers)
                    return response, target_model, key
                elif response.status_code == 429:
                    groq_pool.record_cooldown(key, target_model)
                    groq_pool.sync_headers(key, target_model, response.headers)
                    if not groq_body.get("stream", False):
                        await response.aclose()
                else:
                    if not groq_body.get("stream", False):
                        await response.aclose()
            except Exception:
                groq_pool.record_cooldown(key, target_model)

    return None, None, None