import os
import sys
import hmac
import hashlib
import html as html_lib
import json
import time
import uuid
import asyncio
import tiktoken
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException, Depends, Form, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv

load_dotenv()

# Vercel: 确保同目录模块可被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_catalog import (
    AGNES_IMAGE_MODEL,
    AGNES_MODELS,
    AGNES_TEXT_MODEL,
    AGNES_VIDEO_MODEL,
    CEREBRAS_MODELS,
    GEMMA_MODEL,
    GLM_MODEL,
    GPT_MODEL,
    GROQ_MODELS,
    MODEL_CATALOG,
    get_model_spec,
)
from groq_provider import groq_pool
from agnes_provider import (
    get_agnes_counts,
    get_agnes_metrics,
)
from provider_adapters import AgnesAdapter, GroqAdapter
from access_control import ClientPrincipal, access_manager
from distributed_limits import admit_fixed_window

VERSION = "2.1.1-FastAPI"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

DEFAULT_MODEL = GPT_MODEL

KEY_COOLDOWN = 60

THINKING_MODE = os.getenv("THINKING_MODE", "auto").lower()
MODEL_FALLBACK_MODE = os.getenv("MODEL_FALLBACK_MODE", "auto").lower()

STATS_FILE = "/tmp/gateway_stats.json" if os.path.exists("/tmp") else "gateway_stats.json"
POOL_FILE = "/tmp/gateway_pool.json" if os.path.exists("/tmp") else "gateway_pool.json"

DEBUG_MAX_TEXT_LEN = 20000
DEBUG_CAPTURE_PAYLOADS = os.getenv("DEBUG_CAPTURE_PAYLOADS", "false").lower() == "true"

UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()

CUSTOM_API_KEYS = set(filter(None, os.getenv("CUSTOM_API_KEYS", "").split(",")))
CEREBRAS_API_KEYS = list(filter(None, os.getenv("CEREBRAS_API_KEYS", "").split(",")))

if not CEREBRAS_API_KEYS:
    raise Exception("No CEREBRAS_API_KEYS found in environment variables.")

VALID_DEBUG_PASSWORDS = {k + k for k in CUSTOM_API_KEYS} if CUSTOM_API_KEYS else set()

async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0, connect=10.0),
    limits=httpx.Limits(max_keepalive_connections=50, max_connections=200, keepalive_expiry=60.0),
    http2=True
)

groq_adapter = GroqAdapter(
    redis_url=UPSTASH_REDIS_REST_URL,
    redis_token=UPSTASH_REDIS_REST_TOKEN,
)
agnes_adapter = AgnesAdapter()

async def upstash_get_async(key: str) -> Optional[Any]:
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        return None
    try:
        url = f"{UPSTASH_REDIS_REST_URL}/get/{key}"
        headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
        r = await async_client.get(url, headers=headers, timeout=3.0)
        if r.status_code == 200:
            res = r.json()
            if res.get("result"):
                return json.loads(res["result"])
    except Exception:
        pass
    return None

async def upstash_get_strict_async(key: str) -> Optional[Any]:
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        return None
    url = f"{UPSTASH_REDIS_REST_URL}/get/{key}"
    headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
    response = await async_client.get(url, headers=headers, timeout=3.0)
    response.raise_for_status()
    result = response.json().get("result")
    return json.loads(result) if result else None

async def upstash_set_async(key: str, value: Any) -> bool:
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        return False
    try:
        url = f"{UPSTASH_REDIS_REST_URL}/set/{key}"
        headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
        r = await async_client.post(url, headers=headers, content=json.dumps(value), timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False

async def upstash_pipeline_async(commands: list) -> Optional[list]:
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        return None
    try:
        response = await async_client.post(
            f"{UPSTASH_REDIS_REST_URL}/pipeline",
            headers={"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"},
            json=commands,
            timeout=5.0,
        )
        response.raise_for_status()
        data = response.json()
        return [item.get("result") if isinstance(item, dict) else item for item in data]
    except Exception:
        return None

RUNTIME_CONFIG_KEY = "gateway_runtime_config"
REQUEST_LOGS_KEY = "gateway_request_logs_v2"
DEBUG_LOGS_KEY = "gateway_debug_logs_v2"
runtime_config_lock = asyncio.Lock()
runtime_config_last_refresh = 0.0

async def refresh_runtime_config_async(force: bool = False):
    global THINKING_MODE, MODEL_FALLBACK_MODE, runtime_config_last_refresh
    if not UPSTASH_REDIS_REST_URL:
        return
    now = time.monotonic()
    async with runtime_config_lock:
        if not force and now - runtime_config_last_refresh < 5.0:
            return
        runtime_config_last_refresh = now
    stored = await upstash_get_async(RUNTIME_CONFIG_KEY)
    if not isinstance(stored, dict):
        return
    thinking = str(stored.get("thinking_mode", "")).lower()
    fallback = str(stored.get("fallback_mode", "")).lower()
    if thinking in {"auto", "on", "off"}:
        THINKING_MODE = thinking
    if fallback in {"auto", "off", "force_gpt"}:
        MODEL_FALLBACK_MODE = fallback

async def save_runtime_config_async() -> bool:
    return await upstash_set_async(RUNTIME_CONFIG_KEY, {
        "thinking_mode": THINKING_MODE,
        "fallback_mode": MODEL_FALLBACK_MODE,
    })

async def load_log_deque_async(redis_key: str, target: deque):
    results = await upstash_pipeline_async([["LRANGE", redis_key, 0, target.maxlen - 1]])
    if not results or not isinstance(results[0], list):
        return
    loaded = []
    for value in results[0]:
        try:
            item = json.loads(value)
            if isinstance(item, dict):
                loaded.append(item)
        except Exception:
            pass
    target.clear()
    target.extend(loaded)

async def persist_log_async(redis_key: str, data: dict, maxlen: int):
    await upstash_pipeline_async([
        ["LPUSH", redis_key, json.dumps(data, ensure_ascii=False)],
        ["LTRIM", redis_key, 0, maxlen - 1],
    ])

stats_lock = asyncio.Lock()

def get_default_stats():
    return {
        "total_requests": 0,       
        "fallback_count": 0,       
        "groq_fallback_count": 0,
        "agnes_requests": 0,
        "429_count": 0,            
        "truncated_count": 0,      
        "other_models_count": 0,    
        "models": {m: {"requests": 0, "input_tokens": 0, "output_tokens": 0, "tokens": 0} for m in CEREBRAS_MODELS + GROQ_MODELS + AGNES_MODELS}
    }

GLOBAL_STATS = get_default_stats()

async def init_global_stats():
    global GLOBAL_STATS
    up_stats = await upstash_get_async("gateway_stats")
    if up_stats and "models" in up_stats:
        GLOBAL_STATS.update(up_stats)
        return
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if "models" in saved:
                    GLOBAL_STATS.update(saved)
        except Exception:
            pass

async def save_global_stats_async():
    async with stats_lock:
        if UPSTASH_REDIS_REST_URL:
            await upstash_set_async("gateway_stats", GLOBAL_STATS)
        try:
            with open(STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(GLOBAL_STATS, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

REQUEST_LOGS = deque(maxlen=100)
log_lock = asyncio.Lock()
VIDEO_TASK_OWNERS = {}
video_task_lock = asyncio.Lock()

async def add_log_async(data: dict):
    async with log_lock:
        REQUEST_LOGS.appendleft(data)
    if UPSTASH_REDIS_REST_URL:
        await persist_log_async(REQUEST_LOGS_KEY, data, REQUEST_LOGS.maxlen)

def video_owner_key(task_id: str) -> str:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return f"agnes_video_owner:{digest}"

async def save_video_owner(task_id: str, owner: dict):
    async with video_task_lock:
        VIDEO_TASK_OWNERS[task_id] = owner
    if UPSTASH_REDIS_REST_URL:
        await upstash_set_async(video_owner_key(task_id), owner)

async def get_video_owner(task_id: str) -> Optional[dict]:
    async with video_task_lock:
        owner = VIDEO_TASK_OWNERS.get(task_id)
    if owner:
        return owner
    owner = await upstash_get_async(video_owner_key(task_id))
    if isinstance(owner, dict):
        async with video_task_lock:
            VIDEO_TASK_OWNERS[task_id] = owner
        return owner
    return None

DEBUG_LOGS = deque(maxlen=50)
debug_log_lock = asyncio.Lock()

async def add_debug_log_async(data: dict):
    if not DEBUG_CAPTURE_PAYLOADS:
        data = dict(data)
        data["request_body"] = "[payload capture disabled]"
        data["response_body"] = "[payload capture disabled]"
    async with debug_log_lock:
        DEBUG_LOGS.appendleft(data)
    if UPSTASH_REDIS_REST_URL:
        await persist_log_async(DEBUG_LOGS_KEY, data, DEBUG_LOGS.maxlen)

def truncate_text(text: str, max_len: int = DEBUG_MAX_TEXT_LEN) -> str:
    if isinstance(text, str) and len(text) > max_len:
        return text[:max_len] + f"\n... [已自动截断，原长度: {len(text)} 字符]"
    return text

def estimate_tokens(messages: list = None, text_content: str = "") -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = tiktoken.get_encoding("gpt-4")
    
    if messages:
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
        num_tokens += 2
        return num_tokens
    if text_content:
        return len(encoding.encode(text_content))
    return 0

class AsyncKeyPool:
    def __init__(self, keys: list):
        self.keys = keys
        self.index = 0
        self.lock = asyncio.Lock()
        self.data = {}
        self.last_save_time = 0

    async def init_pool_data(self):
        history_pool = await upstash_get_async("gateway_pool") or {}
        if not history_pool and os.path.exists(POOL_FILE):
            try:
                with open(POOL_FILE, "r", encoding="utf-8") as f:
                    history_pool = json.load(f)
            except Exception:
                pass

        bj_now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")

        for key in self.keys:
            self.data[key] = {}
            stored_key = "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()
            stored_data = history_pool.get(stored_key, history_pool.get(key, {}))
            for model in CEREBRAS_MODELS:
                h_req = stored_data.get(model, {}).get("requests", 0)
                h_tok = stored_data.get(model, {}).get("tokens", 0)
                h_tpd = stored_data.get(model, {}).get("tpd_tokens", 0)
                h_date = stored_data.get(model, {}).get("last_reset_date", "")

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
                    "limit_rpm": 5,
                    "limit_rpd": 2400,
                    "limit_tpm": 30000,
                    "limit_tpd": 1000000,
                    "has_synced": False
                }

    async def save_pool_data_async(self, force: bool = False):
        now = time.time()
        if not force and (now - self.last_save_time < 15):
            return
            
        async with self.lock:
            self.last_save_time = time.time()
            try:
                export = {}
                for k, v in self.data.items():
                    stored_key = "sha256:" + hashlib.sha256(k.encode("utf-8")).hexdigest()
                    export[stored_key] = {}
                    for m, info in v.items():
                        export[stored_key][m] = {
                            "requests": info["requests"],
                            "tokens": info["tokens"],
                            "tpd_tokens": info.get("tpd_tokens", 0),
                            "last_reset_date": info.get("last_reset_date", "")
                        }
                if UPSTASH_REDIS_REST_URL:
                    await upstash_set_async("gateway_pool", export)
                with open(POOL_FILE, "w", encoding="utf-8") as f:
                    json.dump(export, f, ensure_ascii=False, indent=4)
            except Exception:
                pass

    def clean_windows(self, info: dict, now: float):
        while info["req_timestamps"] and now - info["req_timestamps"][0] > 60:
            info["req_timestamps"].popleft()
        while info["token_timestamps"] and now - info["token_timestamps"][0][0] > 60:
            info["token_timestamps"].popleft()

    def get_current_metrics(self, key: str, model: str) -> dict:
        now = time.time()
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

    async def get_next_key_for_request(self, target_model: str, exclude_keys: set = None):
        if exclude_keys is None:
            exclude_keys = set()
            
        now = time.time()
        async with self.lock:
            num_keys = len(self.keys)
            for _ in range(num_keys):
                key = self.keys[self.index]
                self.index = (self.index + 1) % num_keys  
                if key in exclude_keys:
                    continue
                info = self.data[key].get(target_model)
                if not info:
                    continue
                
                self.clean_windows(info, now)
                if info["cooldown"] <= now and len(info["req_timestamps"]) < info["limit_rpm"]:
                    return key, target_model
            return None, None

    async def cooldown_async(self, key: str, model: str):
        async with self.lock:
            if key in self.data and model in self.data[key]:
                self.data[key][model]["cooldown"] = time.time() + KEY_COOLDOWN

    async def success_async(self, key: str, model: str):
        async with self.lock:
            if key in self.data and model in self.data[key]:
                self.data[key][model]["cooldown"] = 0

    async def record_request_attempt_async(self, key: str, model: str, estimated_tokens: int = 0):
        now = time.time()
        async with self.lock:
            if key in self.data and model in self.data[key]:
                info = self.data[key][model]
                info["req_timestamps"].append(now)
                if estimated_tokens > 0:
                    info["token_timestamps"].append((now, estimated_tokens))

    async def sync_headers_async(self, key: str, model: str, headers: httpx.Headers, actual_tokens: int = 0):
        if not key or model not in CEREBRAS_MODELS:
            return
        now = time.time()
        bj_now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")

        try:
            limit_rpd = headers.get("x-ratelimit-limit-requests-day")
            limit_tpm = headers.get("x-ratelimit-limit-tokens-minute")
            rem_rpd = headers.get("x-ratelimit-remaining-requests-day")
            rem_tpm = headers.get("x-ratelimit-remaining-tokens-minute")

            async with self.lock:
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
                async with self.lock:
                    info = self.data[key][model]
                    if info.get("last_reset_date") != bj_now:
                        info["tpd_tokens"] = 0
                        info["last_reset_date"] = bj_now
                    info["tokens"] += actual_tokens
                    info["tpd_tokens"] += actual_tokens
        finally:
            await self.save_pool_data_async()

pool = AsyncKeyPool(CEREBRAS_API_KEYS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_global_stats()
    await pool.init_pool_data()
    await groq_pool.restore_from_upstash(async_client, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN)
    await access_manager.initialize(upstash_get_strict_async)
    await refresh_runtime_config_async(force=True)
    await load_log_deque_async(REQUEST_LOGS_KEY, REQUEST_LOGS)
    await load_log_deque_async(DEBUG_LOGS_KEY, DEBUG_LOGS)
    yield
    await async_client.aclose()

app = FastAPI(title="Cerebras OpenAI Gateway", version=VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sse(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, ensure_ascii=False)
    return f"data: {data}\n\n"

def html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin:0; padding:20px; background-color: #0b0f19; color: #f3f4f6; }}
a {{ color:#3b82f6; text-decoration:none; transition: color 0.2s; }}
a:hover {{ color:#60a5fa; text-decoration:underline; }}
.box {{ max-width: 1200px; margin: 0 auto 20px auto; border: 1px solid #1e293b; border-radius:12px; padding:24px; background: #111827; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }}
h2, h3 {{ color: #ffffff; margin-top:0; border-bottom: 1px solid #1e293b; padding-bottom: 8px; }}
pre, code {{ white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; max-width: 100%; }}
pre {{ overflow-x: auto; line-height: 1.6; }}
.nav {{ max-width: 1200px; margin: 20px auto; font-weight: bold; text-align: center; }}
.nav-btn {{ display: inline-block; padding: 10px 20px; background: #1f2937; color: #fff; border: 1px solid #374151; border-radius: 6px; }}
.nav-btn:hover {{ background: #374151; }}
.grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }}
.card {{ background: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 16px; }}
.metric-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }}
.progress-bar-bg {{ background: #4b5563; height: 6px; border-radius: 3px; overflow: hidden; margin-bottom: 12px; }}
.progress-bar {{ background: #3b82f6; height: 100%; border-radius: 3px; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
.badge-green {{ background: #065f46; color: #34d399; }}
.badge-red {{ background: #991b1b; color: #f87171; }}
.tag {{ color: #9ca3af; font-size: 12px; }}
@media (max-width: 640px) {{
body {{ padding: 10px; }}
.box {{ padding: 14px; border-radius: 8px; }}
h2 {{ font-size: 20px; line-height: 1.3; }}
h3 {{ font-size: 16px; line-height: 1.3; }}
.grid-2 {{ grid-template-columns: minmax(0, 1fr); gap: 12px; }}
.card {{ padding: 12px; }}
.metric-row {{ align-items: flex-start; gap: 8px; }}
.nav-btn {{ width: 100%; box-sizing: border-box; }}
}}
</style>
</head>
<body>
<div class="box">{body}</div>
<div class="nav"><a href="/menu" class="nav-btn">🔙 返回主菜单</a></div>
</body>
</html>"""

async def authenticate_request(request: Request) -> Optional[ClientPrincipal]:
    await access_manager.refresh_if_due(upstash_get_strict_async)
    authorization = request.headers.get("Authorization")
    if not authorization:
        api_key = request.headers.get("X-API-Key") or request.headers.get("api-key")
        if api_key:
            authorization = "Bearer " + api_key.strip()
    return await access_manager.authenticate(authorization)

def api_error(status_code: int, message: str, error_type: str, code: str, request_id: str = "") -> JSONResponse:
    content = {"error": {"message": message, "type": error_type, "param": None, "code": code}}
    if request_id:
        content["request_id"] = request_id
    return JSONResponse(status_code=status_code, content=content)

async def authorize_model_request(
    request: Request, model: str, request_id: str = "", body: Optional[dict] = None
):
    principal = await authenticate_request(request)
    if principal is None:
        return None, api_error(401, "Unauthorized", "auth_error", "unauthorized", request_id)
    if not access_manager.authorize(principal, model):
        return None, api_error(403, f"Model not allowed: {model}", "permission_error", "model_not_allowed", request_id)

    estimated_tokens = estimate_tokens(messages=body.get("messages", [])) if body and body.get("messages") else 0
    quota = await access_manager.consume(
        principal,
        model,
        body=body,
        client=async_client,
        redis_url=UPSTASH_REDIS_REST_URL,
        redis_token=UPSTASH_REDIS_REST_TOKEN,
        estimated_tokens=estimated_tokens,
    )
    if not quota["allowed"]:
        response = api_error(429, "Client provider quota exceeded", "rate_limit_error", "client_quota_exceeded", request_id)
        response.headers["Retry-After"] = str(quota["retry_after"])
        response.headers["X-RateLimit-Limit-Requests"] = str(quota["limit_rpm"])
        response.headers["X-RateLimit-Remaining-Requests"] = "0"
        if quota.get("limit_tpm") is not None:
            response.headers["X-RateLimit-Limit-Tokens"] = str(quota["limit_tpm"])
            response.headers["X-RateLimit-Remaining-Tokens"] = "0"
        return None, response
    return principal, None

def create_admin_session_token() -> str:
    if not ADMIN_API_KEY:
        return ""
    issued_at = str(int(time.time()))
    signature = hmac.new(
        ADMIN_API_KEY.encode("utf-8"),
        ("cpr-admin-session:" + issued_at).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return issued_at + "." + signature

def is_admin_authenticated(request: Request) -> bool:
    supplied = request.cookies.get("admin_session", "")
    issued_at, separator, signature = supplied.partition(".")
    if not ADMIN_API_KEY or not separator or not issued_at.isdigit():
        return False
    if abs(int(time.time()) - int(issued_at)) > 86400:
        return False
    expected = hmac.new(
        ADMIN_API_KEY.encode("utf-8"),
        ("cpr-admin-session:" + issued_at).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

def admin_csrf_token(request: Request) -> str:
    session = request.cookies.get("admin_session", "")
    if not session or not ADMIN_API_KEY:
        return ""
    return hmac.new(
        ADMIN_API_KEY.encode("utf-8"),
        ("cpr-csrf:" + session).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

def admin_required_response() -> HTMLResponse:
    return HTMLResponse(
        content=html_page("Admin required", "<h2>需要管理员权限</h2><p>请先前往 <a href='/admin'>/admin</a> 登录。</p>"),
        status_code=401,
    )

def get_thinking_display(body: dict) -> bool:
    if "thinkingdisplay" in body:
        v = body["thinkingdisplay"]
        return v.lower() == "true" if isinstance(v, str) else bool(v)
    thinking = body.get("thinking", {})
    if isinstance(thinking, dict) and "display" in thinking:
        return bool(thinking["display"])
    return False

def final_thinking(body: dict) -> bool:
    if THINKING_MODE == "on": return True
    if THINKING_MODE == "off": return False
    return get_thinking_display(body)

def sanitize_messages(messages: list, show: bool) -> list:
    result = []
    for m in messages:
        if not isinstance(m, dict): continue
        item = {"role": m.get("role", "user"), "content": m.get("content", "")}
        if show:
            for key in ["reasoning_content", "thinking", "analysis", "reasoning"]:
                if key in m: item[key] = m[key]
        result.append(item)
    return result

def validate_chat_images(model: str, messages: list) -> Optional[dict]:
    spec = get_model_spec(model)
    for message in messages or []:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            is_image_file = block_type == "file" and str(block.get("media_type", block.get("mime_type", ""))).startswith("image/")
            if block_type not in {"image", "image_url", "input_image"} and not is_image_file:
                continue
            image = block.get("image_url", block.get("image", block.get("source", "")))
            if isinstance(image, dict):
                url = image.get("url") or image.get("path") or image.get("file_path") or image.get("filename") or ""
            else:
                url = str(image or block.get("path") or block.get("file_path") or block.get("filename") or "")
            normalized = url.strip().lower()
            is_local = (
                normalized.startswith("file:")
                or (len(url) > 2 and url[1:3] in {":\\", ":/"})
                or url.startswith("\\\\")
            )
            if is_local:
                return {
                    "message": "The gateway cannot access an image path on the client device. Upload the image through a public HTTPS URL or a data:image Base64 URI.",
                    "code": "local_image_unavailable",
                }
            if spec is None or not spec.supports_image_input:
                return {
                    "message": f"Model {model} does not support image input. Use gemma-4-31b or agnes/agnes-2.5-flash for image understanding.",
                    "code": "image_input_not_supported",
                }
    return None

def sanitize_body(body: dict, show_thinking: bool, target_model: str = None) -> dict:
    model = target_model or body.get("model", DEFAULT_MODEL)
    new = {
        "model": model,
        "messages": sanitize_messages(body.get("messages", []), show_thinking),
        "stream": bool(body.get("stream", False))
    }
    if new["stream"]:
        new["stream_options"] = {"include_usage": True}
    
    if not show_thinking:
        model_lower = str(model).lower()
        if "glm" in model_lower:
            new["reasoning_effort"] = "none"
        else:
            new["reasoning_format"] = "hidden"

    for k in ["temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty"]:
        if k in body: new[k] = body[k]
        
    if target_model == GLM_MODEL and "max_completion_tokens" in body:
        if body["max_completion_tokens"] > 8192:
            new["max_completion_tokens"] = 8192

    if "extra_body" in body and isinstance(body["extra_body"], dict):
        new.update(body["extra_body"])
    return new

def remove_thinking(obj: dict) -> dict:
    try:
        for choice in obj.get("choices", []):
            for target in [choice.get("delta", {}), choice.get("message", {})]:
                if isinstance(target, dict):
                    for k in ["reasoning_content", "thinking", "analysis", "reasoning"]:
                        target.pop(k, None)
    except Exception:
        pass
    return obj

async def add_global_usage_async(model: str, usage: dict):
    if not usage:
        return
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", prompt + completion)

    async with stats_lock:
        if model not in GLOBAL_STATS["models"]:
            GLOBAL_STATS["models"][model] = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "tokens": 0}
        s = GLOBAL_STATS["models"][model]
        s["requests"] += 1
        s["input_tokens"] += prompt
        s["output_tokens"] += completion
        s["tokens"] += total
    await save_global_stats_async()

async def finish_log_async(request_id: str, request_model: str, key: str, final_model: str, fallback: bool, start: float, usage: dict = None):
    cost = round(time.time() - start, 2)
    tokens = usage.get("total_tokens", 0) if usage else 0
    k_last = key[-4:] if key else "-"
    bj_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    
    item = {
        "time": bj_time, "id": request_id, "model": request_model,
        "key": k_last, "result": "成功", "fallback": final_model if fallback else "",
        "tokens": tokens, "time_cost": cost
    }
    await add_log_async(item)

async def add_chat_debug_async(
    request_id: str, request_model: str, final_model: str, key_suffix: str,
    status_code: int, start: float, raw: dict, response_body: str,
):
    bj_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    await add_debug_log_async({
        "id": request_id,
        "time": bj_time,
        "request_model": request_model,
        "final_model": final_model,
        "key": key_suffix,
        "status_code": status_code,
        "time_cost": round(time.time() - start, 2),
        "request_body": truncate_text(json.dumps(raw, ensure_ascii=False, indent=2)),
        "response_body": truncate_text(response_body),
    })

async def record_client_completion_async(principal: ClientPrincipal, model: str, usage: dict, body: dict):
    await access_manager.record_tokens(
        principal, model, int((usage or {}).get("completion_tokens", 0)), body=body,
        client=async_client, redis_url=UPSTASH_REDIS_REST_URL,
        redis_token=UPSTASH_REDIS_REST_TOKEN,
    )

async def tracked_external_stream(source, principal: ClientPrincipal, model: str, body: dict):
    try:
        async for chunk in source:
            yield chunk
    finally:
        await source.response.aclose()
        usage = source.usage
        if not usage:
            completion_tokens = estimate_tokens(text_content=source.generated_text)
            usage = {"prompt_tokens": 0, "completion_tokens": completion_tokens, "total_tokens": completion_tokens}
        try:
            await asyncio.wait_for(add_global_usage_async(model, usage), timeout=5.0)
            await asyncio.wait_for(record_client_completion_async(principal, model, usage, body), timeout=5.0)
        except Exception:
            pass

@app.post("/v1/chat/completions")
async def chat(request: Request):
    start = time.time()
    request_id = str(uuid.uuid4())[:8]

    # 1. 解析请求体
    try:
        raw = await request.json()
    except Exception:
        raw = {}

    request_model = raw.get("model", DEFAULT_MODEL)
    request_spec = get_model_spec(request_model)
    if request_spec is None:
        return api_error(400, f"Unsupported model: {request_model}", "invalid_request_error", "unsupported_model", request_id)
    if request_spec.operation != "chat":
        return api_error(400, f"Use the {request_spec.operation} endpoint for this model", "invalid_request_error", "wrong_endpoint", request_id)
    image_error = validate_chat_images(request_model, raw.get("messages", []))
    if image_error:
        return api_error(400, image_error["message"], "invalid_request_error", image_error["code"], request_id)
    principal, auth_error = await authorize_model_request(request, request_model, request_id, raw)
    if auth_error:
        return auth_error

    async with stats_lock:
        GLOBAL_STATS["total_requests"] += 1
    await save_global_stats_async()
    await refresh_runtime_config_async()

    show_thinking = final_thinking(raw)

    if request_model in AGNES_MODELS:
        agnes_result = await agnes_adapter.chat(async_client, raw, timeout=120.0)
        if not agnes_result.available:
            await add_chat_debug_async(
                request_id, request_model, "agnes:none", "", 503, start, raw,
                agnes_result.error or "Agnes unavailable",
            )
            return api_error(503, agnes_result.error or "Agnes unavailable", "service_unavailable", "agnes_unavailable", request_id)
        if not 200 <= agnes_result.status_code < 300:
            try:
                await add_chat_debug_async(
                    request_id, request_model, f"agnes:{agnes_result.site}",
                    agnes_result.credential_suffix, agnes_result.status_code, start,
                    raw, agnes_result.response.text,
                )
                return upstream_response(agnes_result.response)
            finally:
                await agnes_result.close()

        if raw.get("stream", False):
            async def agnes_stream_gen():
                source = None
                try:
                    source = agnes_result.stream(request_id)
                    async for chunk in tracked_external_stream(source, principal, request_model, raw):
                        yield chunk
                finally:
                    await agnes_result.close()
                    await finish_log_async(request_id, request_model, agnes_result.credential_suffix, f"agnes:{agnes_result.site}", False, start)
                    generated = source.generated_text if source else ""
                    samples = "".join(source.sample_chunks) if source else ""
                    await add_chat_debug_async(
                        request_id, request_model, f"agnes:{agnes_result.site}",
                        agnes_result.credential_suffix, 200, start, raw,
                        f"【流式输出】:\n{generated}\n\n【SSE Chunk 样例】:\n{samples}",
                    )

            return StreamingResponse(agnes_stream_gen(), media_type="text/event-stream")

        try:
            result = agnes_result.json(request_id)
            usage = result.get("usage", {})
            await add_global_usage_async(request_model, usage)
            await record_client_completion_async(principal, request_model, usage, raw)
            await finish_log_async(request_id, request_model, agnes_result.credential_suffix, f"agnes:{agnes_result.site}", False, start, usage)
            await add_chat_debug_async(
                request_id, request_model, f"agnes:{agnes_result.site}",
                agnes_result.credential_suffix, agnes_result.status_code, start,
                raw, json.dumps(result, ensure_ascii=False, indent=2),
            )
            return JSONResponse(status_code=agnes_result.status_code, content=result)
        finally:
            await agnes_result.close()

    if request_model in GROQ_MODELS:
        groq_result = await groq_adapter.chat(async_client, raw, show_thinking)

        if groq_result.available:
            if raw.get("stream", False):
                async def direct_groq_stream_gen():
                    try:
                        source = groq_result.stream(request_id)
                        async for chunk in tracked_external_stream(source, principal, groq_result.model, raw):
                            yield chunk
                    finally:
                        await groq_result.close()
                        await groq_result.record_success()
                        await finish_log_async(request_id, request_model, groq_result.credential_suffix, f"groq:{groq_result.model}", False, start)

                return StreamingResponse(direct_groq_stream_gen(), media_type="text/event-stream")

            try:
                res_data = groq_result.json(request_id)
                usage = res_data.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                await add_global_usage_async(groq_result.model, usage)
                await record_client_completion_async(principal, groq_result.model, usage, raw)
                await groq_result.record_success(total_tokens)
                await finish_log_async(request_id, request_model, groq_result.credential_suffix, f"groq:{groq_result.model}", False, start, usage)

                bj_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
                await add_debug_log_async({
                    "id": request_id, "time": bj_time, "request_model": request_model,
                    "final_model": f"groq:{groq_result.model}", "key": groq_result.credential_suffix, "status_code": 200,
                    "time_cost": round(time.time() - start, 2),
                    "request_body": truncate_text(json.dumps(raw, ensure_ascii=False, indent=2)),
                    "response_body": truncate_text(json.dumps(res_data, ensure_ascii=False, indent=2))
                })
                return JSONResponse(status_code=200, content=res_data)
            finally:
                await groq_result.close()

        return JSONResponse(status_code=503, content={
            "error": {"message": "All Groq keys/models failed", "type": "service_unavailable", "param": None, "code": "groq_failed"},
            "request_id": request_id
        })

    # 4. 尝试 Cerebras 模型池轮询
    models_to_try = [request_model] + [m for m in CEREBRAS_MODELS if m != request_model]
    if MODEL_FALLBACK_MODE == "off":
        models_to_try = [request_model]
    elif MODEL_FALLBACK_MODE == "force_gpt":
        if not access_manager.authorize(principal, GPT_MODEL):
            return api_error(403, f"Model not allowed: {GPT_MODEL}", "permission_error", "model_not_allowed", request_id)
        models_to_try = [GPT_MODEL]
    models_to_try = [model for model in models_to_try if access_manager.authorize(principal, model)]
    
    last_error = None

    for target_model in models_to_try:
        body = sanitize_body(raw, show_thinking, target_model=target_model)
        tried_keys = set()
        max_attempts = len(CEREBRAS_API_KEYS)

        for attempt in range(max_attempts):
            selected_key, selected_model = await pool.get_next_key_for_request(target_model, tried_keys)
            if not selected_key:
                break

            tried_keys.add(selected_key)
            estimated_in = estimate_tokens(messages=raw.get("messages", []))
            body["model"] = selected_model
            admitted = await admit_fixed_window(
                async_client, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN,
                "cerebras", selected_key, selected_model,
                pool.data[selected_key][selected_model]["limit_rpm"],
            )
            if admitted is False:
                last_error = "Distributed upstream quota unavailable"
                continue
            await pool.record_request_attempt_async(selected_key, selected_model, estimated_in)

            fallback_happened = (selected_model != request_model)

            # --- 流式请求处理 ---
            if body.get("stream", False):
                try:
                    req = async_client.build_request(
                        "POST",
                        f"{CEREBRAS_BASE_URL}/chat/completions",
                        json=body,
                        headers={"Authorization": f"Bearer {selected_key}", "Content-Type": "application/json"},
                        timeout=60.0
                    )
                    response = await async_client.send(req, stream=True)

                    # 如果非 200 响应，清理并尝试下一个 Key/Model
                    if response.status_code != 200:
                        if response.status_code == 429:
                            async with stats_lock: GLOBAL_STATS["429_count"] += 1
                            await save_global_stats_async()
                            last_error = f"429 Limit reached on ****{selected_key[-4:]}"
                        else:
                            last_error = f"Upstream error {response.status_code}"

                        await pool.cooldown_async(selected_key, selected_model)
                        await pool.sync_headers_async(selected_key, selected_model, response.headers)
                        await response.aclose()
                        continue

                    # 只有 200 OK 成功建立流连接后才返回 StreamingResponse
                    await pool.sync_headers_async(selected_key, selected_model, response.headers)
                    await pool.success_async(selected_key, selected_model)

                    async def event_generator():
                        last_usage = None
                        generated_text = ""
                        stream_chunks = []
                        is_truncated = False

                        try:
                            async for line in response.aiter_lines():
                                if not line: continue
                                line_str = line.strip()
                                if not line_str.startswith("data:"): continue
                                data_body = line_str[5:].strip()

                                if data_body == "[DONE]":
                                    yield "data: [DONE]\n\n"
                                    break

                                try:
                                    obj = json.loads(data_body)
                                    if "id" not in obj:
                                        obj["id"] = f"chatcmpl-{request_id}"
                                    if "object" not in obj:
                                        obj["object"] = "chat.completion.chunk"
                                    if "created" not in obj:
                                        obj["created"] = int(time.time())
                                    if "model" not in obj:
                                        obj["model"] = selected_model

                                    if "usage" in obj and obj["usage"]:
                                        last_usage = obj["usage"]

                                    for choice in obj.get("choices", []):
                                        if choice.get("finish_reason") == "length":
                                            is_truncated = True
                                        delta = choice.get("delta", {})
                                        if "content" in delta and delta["content"]:
                                            generated_text += delta["content"]

                                    if not show_thinking:
                                        obj = remove_thinking(obj)

                                    chunk_out = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
                                    if len("".join(stream_chunks)) < DEBUG_MAX_TEXT_LEN:
                                        stream_chunks.append(chunk_out)
                                    yield chunk_out
                                except Exception:
                                    yield line_str + "\n\n"

                        finally:
                            # 保证流结束或异常断开时能正确收尾
                            if is_truncated:
                                async with stats_lock: GLOBAL_STATS["truncated_count"] += 1
                                await save_global_stats_async()

                            if not last_usage:
                                out_tokens = estimate_tokens(text_content=generated_text)
                                last_usage = {"prompt_tokens": estimated_in, "completion_tokens": out_tokens, "total_tokens": estimated_in + out_tokens}

                            await add_global_usage_async(selected_model, last_usage)
                            await record_client_completion_async(principal, selected_model, last_usage, raw)
                            await pool.sync_headers_async(selected_key, selected_model, response.headers, actual_tokens=last_usage.get("total_tokens", 0))

                            if fallback_happened:
                                async with stats_lock: GLOBAL_STATS["fallback_count"] += 1
                                await save_global_stats_async()

                            await finish_log_async(request_id, request_model, selected_key, selected_model, fallback_happened, start, last_usage)

                            bj_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
                            trunc_flag = " [⚠️ 长度截断]" if is_truncated else ""
                            resp_debug_str = f"【流式输出】{trunc_flag}:\n{generated_text}\n\n【SSE Chunk 样例】:\n" + "".join(stream_chunks[:10])
                            await add_debug_log_async({
                                "id": request_id, "time": bj_time, "request_model": request_model,
                                "final_model": selected_model, "key": selected_key[-4:], "status_code": 200,
                                "time_cost": round(time.time() - start, 2),
                                "request_body": truncate_text(json.dumps(raw, ensure_ascii=False, indent=2)),
                                "response_body": truncate_text(resp_debug_str)
                            })
                            await response.aclose()

                    return StreamingResponse(event_generator(), media_type="text/event-stream")

                except Exception as e:
                    await pool.cooldown_async(selected_key, selected_model)
                    last_error = str(e)
                    continue

            # --- 非流式请求处理 ---
            try:
                r = await async_client.post(
                    f"{CEREBRAS_BASE_URL}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {selected_key}", "Content-Type": "application/json"},
                    timeout=60.0
                )

                if r.status_code == 200:
                    result = r.json()
                    await pool.success_async(selected_key, selected_model)
                    is_truncated = any(choice.get("finish_reason") == "length" for choice in result.get("choices", []))

                    if is_truncated:
                        async with stats_lock: GLOBAL_STATS["truncated_count"] += 1
                        await save_global_stats_async()

                    usage = result.get("usage")
                    if not usage:
                        text_out = "".join(
                            choice.get("message", {}).get("content", "")
                            for choice in result.get("choices", [])
                            if choice.get("message", {}).get("content")
                        )
                        out_tokens = estimate_tokens(text_content=text_out)
                        usage = {"prompt_tokens": estimated_in, "completion_tokens": out_tokens, "total_tokens": estimated_in + out_tokens}
                        result["usage"] = usage

                    await add_global_usage_async(selected_model, usage)
                    await record_client_completion_async(principal, selected_model, usage, raw)
                    await pool.sync_headers_async(selected_key, selected_model, r.headers, actual_tokens=usage.get("total_tokens", 0))

                    if not show_thinking:
                        result = remove_thinking(result)

                    if fallback_happened:
                        async with stats_lock: GLOBAL_STATS["fallback_count"] += 1
                        await save_global_stats_async()

                    await finish_log_async(request_id, request_model, selected_key, selected_model, fallback_happened, start, usage)

                    bj_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
                    await add_debug_log_async({
                        "id": request_id, "time": bj_time, "request_model": request_model,
                        "final_model": selected_model, "key": selected_key[-4:], "status_code": 200,
                        "time_cost": round(time.time() - start, 2),
                        "request_body": truncate_text(json.dumps(raw, ensure_ascii=False, indent=2)),
                        "response_body": truncate_text(json.dumps(result, ensure_ascii=False, indent=2))
                    })

                    return JSONResponse(status_code=200, content=result)

                elif r.status_code == 429:
                    async with stats_lock: GLOBAL_STATS["429_count"] += 1
                    await save_global_stats_async()
                    await pool.cooldown_async(selected_key, selected_model)
                    await pool.sync_headers_async(selected_key, selected_model, r.headers)
                    last_error = f"429 Limit reached on ****{selected_key[-4:]}"
                    continue
                else:
                    last_error = f"Upstream error {r.status_code}"
                    continue

            except Exception as e:
                await pool.cooldown_async(selected_key, selected_model)
                last_error = str(e)
                continue

    # 5. Cerebras 全部失败后，降级尝试 Groq
    allowed_groq_models = [model for model in GROQ_MODELS if access_manager.authorize(principal, model)]
    groq_result = None
    if MODEL_FALLBACK_MODE == "auto":
        if allowed_groq_models and access_manager.provider_for_model(request_model) != "groq":
            fallback_quota = await access_manager.consume(
                principal, allowed_groq_models[0], body=raw,
                client=async_client, redis_url=UPSTASH_REDIS_REST_URL,
                redis_token=UPSTASH_REDIS_REST_TOKEN,
                estimated_tokens=estimate_tokens(messages=raw.get("messages", [])),
            )
            if not fallback_quota["allowed"]:
                return api_error(429, "Groq client quota exceeded", "rate_limit_error", "client_quota_exceeded", request_id)
        for fallback_model in allowed_groq_models:
            groq_raw = dict(raw)
            groq_raw["model"] = fallback_model
            groq_result = await groq_adapter.chat(async_client, groq_raw, show_thinking)
            if groq_result.available:
                break

    if groq_result and groq_result.available:
        async with stats_lock:
            GLOBAL_STATS["groq_fallback_count"] += 1
            GLOBAL_STATS["fallback_count"] += 1
        await save_global_stats_async()

        if raw.get("stream", False):
            async def groq_stream_gen():
                try:
                    source = groq_result.stream(request_id)
                    async for chunk in tracked_external_stream(source, principal, groq_result.model, raw):
                        yield chunk
                finally:
                    await groq_result.close()
                    await groq_result.record_success()
                    await finish_log_async(request_id, request_model, groq_result.credential_suffix, f"groq:{groq_result.model}", True, start)

            return StreamingResponse(groq_stream_gen(), media_type="text/event-stream")
        else:
            try:
                res_data = groq_result.json(request_id)
                usage = res_data.get("usage", {})
                tot_tok = usage.get("total_tokens", 0)
                await add_global_usage_async(groq_result.model, usage)
                await record_client_completion_async(principal, groq_result.model, usage, raw)
                await groq_result.record_success(tot_tok)
                await finish_log_async(request_id, request_model, groq_result.credential_suffix, f"groq:{groq_result.model}", True, start, usage)

                bj_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
                await add_debug_log_async({
                    "id": request_id, "time": bj_time, "request_model": request_model,
                    "final_model": f"groq:{groq_result.model}", "key": groq_result.credential_suffix, "status_code": 200,
                    "time_cost": round(time.time() - start, 2),
                    "request_body": truncate_text(json.dumps(raw, ensure_ascii=False, indent=2)),
                    "response_body": truncate_text(json.dumps(res_data, ensure_ascii=False, indent=2))
                })
                return JSONResponse(status_code=200, content=res_data)
            finally:
                await groq_result.close()

    # 6. 所有 Provider、模型、Key 均失败
    error_msg = f"All Cerebras and Groq keys/models failed. Last error: {last_error}"
    bj_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    await add_debug_log_async({
        "id": request_id, "time": bj_time, "request_model": request_model,
        "final_model": "None", "key": "None", "status_code": 503,
        "time_cost": round(time.time() - start, 2),
        "request_body": truncate_text(json.dumps(raw, ensure_ascii=False, indent=2)),
        "response_body": truncate_text(error_msg)
    })

    return JSONResponse(status_code=503, content={
        "error": {"message": error_msg, "type": "service_unavailable", "param": None, "code": "all_providers_failed"},
        "request_id": request_id
    })

def upstream_response(response: httpx.Response) -> Response:
    content_type = response.headers.get("content-type", "application/json").split(";", 1)[0]
    return Response(content=response.content, status_code=response.status_code, media_type=content_type)

def provider_json_response(result) -> Response:
    try:
        return JSONResponse(content=result.json(), status_code=result.status_code)
    except Exception:
        return upstream_response(result.response)

@app.post("/v1/images/generations")
async def agnes_images(request: Request):
    request_id = str(uuid.uuid4())[:8]
    try:
        body = await request.json()
    except Exception:
        return api_error(400, "Invalid JSON body", "invalid_request_error", "invalid_json", request_id)

    model = body.get("model", AGNES_IMAGE_MODEL)
    if model != AGNES_IMAGE_MODEL:
        return api_error(400, f"Unsupported image model: {model}", "invalid_request_error", "unsupported_model", request_id)
    _, auth_error = await authorize_model_request(request, model, request_id, body)
    if auth_error:
        return auth_error

    result = await agnes_adapter.request(
        async_client, "POST", "/images/generations", body=body,
        public_model=model, timeout=360.0,
    )
    if not result.available:
        return api_error(503, result.error or "Agnes image service unavailable", "service_unavailable", "agnes_unavailable", request_id)
    try:
        if not 200 <= result.status_code < 300:
            return upstream_response(result.response)
        async with stats_lock:
            GLOBAL_STATS["agnes_requests"] = GLOBAL_STATS.get("agnes_requests", 0) + 1
        await save_global_stats_async()
        return provider_json_response(result)
    finally:
        await result.close()

@app.post("/v1/videos")
async def agnes_videos(request: Request):
    request_id = str(uuid.uuid4())[:8]
    try:
        body = await request.json()
    except Exception:
        return api_error(400, "Invalid JSON body", "invalid_request_error", "invalid_json", request_id)

    model = body.get("model", AGNES_VIDEO_MODEL)
    if model != AGNES_VIDEO_MODEL:
        return api_error(400, f"Unsupported video model: {model}", "invalid_request_error", "unsupported_model", request_id)
    principal, auth_error = await authorize_model_request(request, model, request_id, body)
    if auth_error:
        return auth_error
    if principal.is_open:
        return api_error(401, "Agnes video endpoints require a configured client key", "auth_error", "video_auth_required", request_id)

    result = await agnes_adapter.request(
        async_client, "POST", "/videos", body=body,
        public_model=model, timeout=120.0,
    )
    if not result.available:
        return api_error(503, result.error or "Agnes video service unavailable", "service_unavailable", "agnes_unavailable", request_id)
    try:
        if not 200 <= result.status_code < 300:
            return upstream_response(result.response)
        try:
            result_data = result.json()
        except Exception:
            result_data = {}
        owner_data = {
            "client_id": principal.client_id,
            "affinity_id": result.route_id,
        }
        task_identifiers = {
            str(value) for value in (
                result_data.get("video_id"), result_data.get("task_id"), result_data.get("id")
            ) if value
        }
        for task_identifier in task_identifiers:
            await save_video_owner(task_identifier, owner_data)
        async with stats_lock:
            GLOBAL_STATS["agnes_requests"] = GLOBAL_STATS.get("agnes_requests", 0) + 1
        await save_global_stats_async()
        return provider_json_response(result)
    finally:
        await result.close()

@app.get("/v1/videos/{task_id}")
@app.get("/agnesapi")
async def agnes_video_result(request: Request, task_id: Optional[str] = None, video_id: Optional[str] = None):
    lookup_id = video_id or task_id
    if not lookup_id:
        return api_error(400, "video_id or task_id is required", "invalid_request_error", "missing_video_id")

    principal = await authenticate_request(request)
    if principal is None:
        return api_error(401, "Unauthorized", "auth_error", "unauthorized")
    if principal.is_open:
        return api_error(401, "Agnes video endpoints require a configured client key", "auth_error", "video_auth_required")
    if not access_manager.authorize(principal, AGNES_VIDEO_MODEL):
        return api_error(403, "Model not allowed", "permission_error", "model_not_allowed")

    owner = await get_video_owner(lookup_id)
    if not owner:
        return api_error(404, "Video task not found", "invalid_request_error", "video_task_not_found")
    if owner.get("client_id") != principal.client_id:
        return api_error(403, "Video task not owned by this key", "permission_error", "video_task_forbidden")

    query = dict(request.query_params)
    query.pop("video_id", None)
    result = await agnes_adapter.query_video(
        async_client, lookup_id, query=query, timeout=60.0,
        affinity_id=owner.get("affinity_id"),
        candidate_index=owner.get("candidate_index"),
    )
    if not result.available:
        return api_error(503, result.error or "Agnes video result unavailable", "service_unavailable", "agnes_unavailable")
    try:
        return provider_json_response(result)
    finally:
        await result.close()

@app.get("/menu", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def menu():
    await refresh_runtime_config_async()
    body = f"""<h2>🧠 Cerebras OpenAI Gateway Menu</h2>
<p><strong>Version:</strong> {VERSION}</p>
<p style="color: #9ca3af; font-size: 14px; margin-top: -10px;">作者：速冻月饼 | 🔗 <a href="https://github.com/xyrct301/cerebras-proxy-re" target="_blank">GitHub 开源仓库</a></p>
<hr style="border-color:#1e293b;"/>
<h3>📌 API Endpoint</h3>
<p>🔗 <a href="/v1/models" target="_blank">/v1/models (查看可用模型列表)</a></p>
<p> <code>POST /v1/chat/completions</code> (标准 OpenAI 格式网关)</p>
<p> <code>POST /v1/images/generations</code> (Agnes 图片生成)</p>
<p> <code>POST /v1/videos</code> (Agnes 视频任务)</p>
<h3>📊 监控中心 (Monitor)</h3>
<p>📈 <a href="/status">/status (实时上游限额 & 物理Key高级看板)</a></p>
<p>📜 <a href="/log">/log (最近 100 条请求历史)</a></p>
<p>🔍 <a href="/debug">/debug (最近 50 条全量请求体/响应体深度调试)</a></p>
<p>⚙️ <a href="/config">/config (系统核心配置)</a></p>
<p>🔐 <a href="/admin">/admin (客户端 Key 与贡献额度管理)</a></p>
<p>❤️ <a href="/health">/health (微服务健康检查)</a></p>
<h3>🎛️ 控制策略 (Control Center)</h3>
<p>⚙️ <a href="/thinkingdisplay">/thinkingdisplay (深度思考输出控制: {THINKING_MODE.upper()})</a></p>
<p>🔀 <a href="/fallbackmode">/fallbackmode (模型降级策略控制: {MODEL_FALLBACK_MODE.upper()})</a></p>
<h3>🤖 托管模型矩阵</h3>
<pre style="background:#1f2937; padding:12px; border-radius:6px; border:1px solid #374151;">
Cerebras ({len(CEREBRAS_MODELS)}): {', '.join(CEREBRAS_MODELS)}
Groq ({len(GROQ_MODELS)}): {', '.join(GROQ_MODELS)}
Agnes ({len(AGNES_MODELS)}): {', '.join(AGNES_MODELS)}
</pre>"""
    return HTMLResponse(content=html_page("Menu", body))

@app.get("/status", response_class=HTMLResponse)
async def status(request: Request):
    if not is_admin_authenticated(request):
        return admin_required_response()
    now = time.time()
    agnes_metrics = await get_agnes_metrics()
    
    # --- Cerebras 限额统计 ---
    cerebras_global_limits = {m: {"rpm": 0, "rpd": 0, "tpm": 0, "tpd": 0, "cur_rpm": 0, "cur_rpd": 0, "cur_tpm": 0, "cur_tpd": 0} for m in CEREBRAS_MODELS}
    for key in CEREBRAS_API_KEYS:
        for model in CEREBRAS_MODELS:
            metrics = pool.get_current_metrics(key, model)
            cerebras_global_limits[model]["rpm"] += metrics["limit_rpm"]
            cerebras_global_limits[model]["rpd"] += metrics["limit_rpd"]
            cerebras_global_limits[model]["tpm"] += metrics["limit_tpm"]
            cerebras_global_limits[model]["tpd"] += metrics["limit_tpd"]
            cerebras_global_limits[model]["cur_rpm"] += metrics["current_rpm"]
            cerebras_global_limits[model]["cur_rpd"] += metrics["current_rpd"]
            cerebras_global_limits[model]["cur_tpm"] += metrics["current_tpm"]
            cerebras_global_limits[model]["cur_tpd"] += metrics["current_tpd"]

    # --- Groq 限额统计 ---
    groq_global_limits = {m: {"rpm": 0, "rpd": 0, "tpm": 0, "tpd": 0, "cur_rpm": 0, "cur_rpd": 0, "cur_tpm": 0, "cur_tpd": 0} for m in GROQ_MODELS}
    for key in groq_pool.keys:
        for model in GROQ_MODELS:
            metrics = groq_pool.get_current_metrics(key, model)
            groq_global_limits[model]["rpm"] += metrics["limit_rpm"]
            groq_global_limits[model]["rpd"] += metrics["limit_rpd"]
            groq_global_limits[model]["tpm"] += metrics["limit_tpm"]
            groq_global_limits[model]["tpd"] += metrics["limit_tpd"]
            groq_global_limits[model]["cur_rpm"] += metrics["current_rpm"]
            groq_global_limits[model]["cur_rpd"] += metrics["current_rpd"]
            groq_global_limits[model]["cur_tpm"] += metrics["current_tpm"]
            groq_global_limits[model]["cur_tpd"] += metrics["current_tpd"]

    def render_progress(cur, limit):
        pct = min(100, round((cur / limit * 100), 1)) if limit > 0 else 0
        color = "#ef4444" if pct > 80 else ("#eab308" if pct > 50 else "#3b82f6")
        return f"""
        <div class="metric-row"><span>进度: {cur} / {limit}</span><span>{pct}%</span></div>
        <div class="progress-bar-bg"><div class="progress-bar" style="width: {pct}%; background: {color};"></div></div>
        """

    html = f"""
    <h2>📊 Gateway 实时监控看板 <span class="tag">v{VERSION}</span></h2>
    <div class="grid-2" style="grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top:15px;">
        <div class="card">
            <div class="tag">总进站请求</div>
            <div style="font-size:28px; font-weight:bold; margin-top:4px;">{GLOBAL_STATS['total_requests']}</div>
        </div>
        <div class="card">
            <div class="tag">降级接管 (Fallback)</div>
            <div style="font-size:28px; font-weight:bold; color:#eab308; margin-top:4px;">{GLOBAL_STATS['fallback_count']}</div>
        </div>
        <div class="card">
            <div class="tag">⚡ Groq 应急降级计数</div>
            <div style="font-size:28px; font-weight:bold; color:#38bdf8; margin-top:4px;">{GLOBAL_STATS.get('groq_fallback_count', 0)}</div>
        </div>
        <div class="card">
            <div class="tag">🚨 触发长度截断</div>
            <div style="font-size:28px; font-weight:bold; color:#f97316; margin-top:4px;">{GLOBAL_STATS.get('truncated_count', 0)}</div>
        </div>
        <div class="card">
            <div class="tag">上游 429 阻断</div>
            <div style="font-size:28px; font-weight:bold; color:#ef4444; margin-top:4px;">{GLOBAL_STATS['429_count']}</div>
        </div>
    </div>
    
    <h3>🔮 Cerebras 核心模型额度汇总</h3>
    <div class="grid-2">
    """
    
    for model in CEREBRAS_MODELS:
        g_data = GLOBAL_STATS["models"].get(model, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "tokens": 0})
        g_lim = cerebras_global_limits[model]
        html += f"""
        <div class="card">
            <div style="font-size:16px; font-weight:bold; color:#60a5fa; margin-bottom:12px;">📌 模型: {model}</div>
            <div class="metric-row"><span class="tag">累计处理请求:</span> <strong>{g_data['requests']}</strong></div>
            <div class="metric-row"><span class="tag">输入 / 输出 Tokens:</span> <span>{g_data['input_tokens']} / {g_data['output_tokens']}</span></div>
            <div class="metric-row" style="border-bottom:1px solid #374151; padding-bottom:8px; margin-bottom:12px;"><span class="tag">历史总耗费 Tokens:</span> <strong style="color:#10b981;">{g_data['tokens']}</strong></div>
            <div class="tag">RPM (每分钟请求数)</div>{render_progress(g_lim['cur_rpm'], g_lim['rpm'])}
            <div class="tag">RPD (每日请求数)</div>{render_progress(g_lim['cur_rpd'], g_lim['rpd'])}
            <div class="tag">TPM (每分钟Tokens)</div>{render_progress(g_lim['cur_tpm'], g_lim['tpm'])}
            <div class="tag">TPD (每日Tokens)</div>{render_progress(g_lim['cur_tpd'], g_lim['tpd'])}
        </div>
        """
    html += "</div>"

    # === Groq 托管模型全局汇总 ===
    html += "<h3>⚡ Groq 托管模型全局汇总</h3><div class='grid-2'>"
    for gm in GROQ_MODELS:
        g_data = GLOBAL_STATS["models"].get(gm, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "tokens": 0})
        g_lim = groq_global_limits[gm]
        html += f"""
        <div class="card" style="border-left: 4px solid #38bdf8;">
            <div style="font-size:16px; font-weight:bold; color:#38bdf8; margin-bottom:12px;">🤖 Groq: {gm}</div>
            <div class="metric-row"><span class="tag">累计处理请求:</span> <strong>{g_data['requests']}</strong></div>
            <div class="metric-row"><span class="tag">输入 / 输出 Tokens:</span> <span>{g_data['input_tokens']} / {g_data['output_tokens']}</span></div>
            <div class="metric-row" style="border-bottom:1px solid #374151; padding-bottom:8px; margin-bottom:12px;"><span class="tag">历史总耗费 Tokens:</span> <strong style="color:#10b981;">{g_data['tokens']}</strong></div>
            <div class="tag">RPM (每分钟请求数)</div>{render_progress(g_lim['cur_rpm'], g_lim['rpm'])}
            <div class="tag">RPD (每日请求数)</div>{render_progress(g_lim['cur_rpd'], g_lim['rpd'])}
            <div class="tag">TPM (每分钟Tokens)</div>{render_progress(g_lim['cur_tpm'], g_lim['tpm'])}
            <div class="tag">TPD (每日Tokens)</div>{render_progress(g_lim['cur_tpd'], g_lim['tpd'])}
        </div>
        """
    html += "</div>"

    # === Cerebras 物理 API-Key 状态细节矩阵 ===
    html += "<h3>🔑 Cerebras API-Key 细节矩阵</h3><div class='grid-2'>"
    for key in CEREBRAS_API_KEYS:
        k_suffix = key[-4:] if len(key) > 4 else key
        html += f"""<div class="card" style="border-left: 4px solid #3b82f6;"><div style="font-weight:bold; font-size:15px; margin-bottom:12px;">🗝️ Key: ****{k_suffix}</div>"""
        for model in CEREBRAS_MODELS:
            info = pool.data[key][model]
            cd = max(0, int(info["cooldown"] - now))
            status_badge = f'<span class="badge badge-red">🔴 冷却 ({cd}s)</span>' if cd > 0 else '<span class="badge badge-green">🟢 正常</span>'
            metrics = pool.get_current_metrics(key, model)
            html += f"""
            <div style="background:#111827; padding:10px; border-radius:6px; margin-bottom:10px; border:1px solid #374151;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-size:12px; font-weight:bold; color:#9ca3af;">🤖 {model}</span>
                    {status_badge}
                </div>
                <div style="font-size:11px; color:#d1d5db; display:grid; grid-template-columns:1fr 1fr; gap:4px;">
                    <div>RPM: {metrics['current_rpm']}/{metrics['limit_rpm']}</div>
                    <div>RPD: {metrics['current_rpd']}/{metrics['limit_rpd']}</div>
                    <div>TPM: {metrics['current_tpm']}/{metrics['limit_tpm']}</div>
                    <div>TPD: {metrics['current_tpd']}/{metrics['limit_tpd']}</div>
                </div>
            </div>
            """
        html += "</div>"
    html += "</div>"

    # === Groq 物理 API-Key 状态细节矩阵 ===
    html += "<h3>⚡ Groq API-Key 细节矩阵</h3><div class='grid-2'>"
    if groq_pool.keys:
        for key in groq_pool.keys:
            k_suffix = key[-4:] if len(key) > 4 else key
            html += f"""<div class="card" style="border-left: 4px solid #38bdf8;"><div style="font-weight:bold; font-size:15px; margin-bottom:12px;">🗝️ Key: ****{k_suffix}</div>"""
            for model in GROQ_MODELS:
                info = groq_pool.data.get(key, {}).get(model, {"cooldown": 0})
                cd = max(0, int(info.get("cooldown", 0) - now))
                status_badge = f'<span class="badge badge-red">🔴 冷却 ({cd}s)</span>' if cd > 0 else '<span class="badge badge-green">🟢 正常</span>'
                metrics = groq_pool.get_current_metrics(key, model)
                html += f"""
                <div style="background:#111827; padding:10px; border-radius:6px; margin-bottom:10px; border:1px solid #374151;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:12px; font-weight:bold; color:#9ca3af;">🤖 {model}</span>
                        {status_badge}
                    </div>
                    <div style="font-size:11px; color:#d1d5db; display:grid; grid-template-columns:1fr 1fr; gap:4px;">
                        <div>RPM: {metrics['current_rpm']}/{metrics['limit_rpm']}</div>
                        <div>RPD: {metrics['current_rpd']}/{metrics['limit_rpd']}</div>
                        <div>TPM: {metrics['current_tpm']}/{metrics['limit_tpm']}</div>
                        <div>TPD: {metrics['current_tpd']}/{metrics['limit_tpd']}</div>
                    </div>
                </div>
                """
            html += "</div>"
    else:
        html += "<p class='tag'>未配置 GROQ_API_KEYS</p>"
    html += "</div>"

    html += "<h3>🌐 Agnes 双站 API-Key 状态</h3><div class='grid-2'>"
    if agnes_metrics["candidates"]:
        for candidate in agnes_metrics["candidates"]:
            status_badge = f'<span class="badge badge-red">冷却 ({candidate["cooldown"]}s)</span>' if candidate["cooldown"] else '<span class="badge badge-green">正常</span>'
            model_lines = "".join(
                f'<div>{html_lib.escape(bucket)}: {data["current_rpm"]}/{data["limit_rpm"]} RPM</div>'
                for bucket, data in candidate["models"].items()
            )
            html += f"""
            <div class="card" style="border-left:4px solid #10b981;">
                <div style="display:flex;justify-content:space-between;gap:8px;"><strong>{candidate['site'].upper()} ****{candidate['key_suffix']}</strong>{status_badge}</div>
                <div class="tag" style="margin:8px 0;">累计尝试: {candidate['request_total']}</div>
                <div style="font-size:11px;display:grid;grid-template-columns:1fr 1fr;gap:5px;">{model_lines}</div>
            </div>"""
    else:
        html += "<p class='tag'>未配置 Agnes API Keys</p>"
    html += "</div>"

    return HTMLResponse(content=html_page("Status 看板", html))

@app.get("/log", response_class=HTMLResponse)
async def log_page(request: Request):
    if not is_admin_authenticated(request):
        return admin_required_response()
    if UPSTASH_REDIS_REST_URL:
        async with log_lock:
            await load_log_deque_async(REQUEST_LOGS_KEY, REQUEST_LOGS)
    lines = ["="*50, "📜 历史回溯请求日志 (最近100条)", "="*50]
    async with log_lock:
        for x in REQUEST_LOGS:
            fallback_text = f" | 🔄 降级至->{x['fallback']}" if x['fallback'] else " | ✅ 成功"
            lines.append(
                f"[{x['time']}] ID:{x['id']} | {x['model']} (Key:****{x['key']}){fallback_text} | Tokens:{x['tokens']} | 耗时:{x['time_cost']}s"
            )
    log_content = "\n".join(lines)
    html_body = f"""
    <h2>📜 历史回溯请求日志</h2>
    <div style="background:#1f2937; color:#f3f4f6; padding:15px; border-radius:6px; border:1px solid #374151; white-space: pre-wrap; word-break: break-all; overflow-x: auto; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; line-height: 1.6;">{log_content}</div>
    """
    return HTMLResponse(content=html_page("Request Logs", html_body))

@app.get("/debug", response_class=HTMLResponse)
async def debug(request: Request):
    if not is_admin_authenticated(request):
        return admin_required_response()

    if UPSTASH_REDIS_REST_URL:
        async with debug_log_lock:
            await load_log_deque_async(DEBUG_LOGS_KEY, DEBUG_LOGS)

    copy_script = """
    <script>
    async function copyDebugInfo(idx) {
        const packageEl = document.getElementById('debug-package-' + idx);
        if (!packageEl) {
            alert('复制失败: 找不到调试包');
            return;
        }
        const text = packageEl.value;
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(text);
            } else {
                packageEl.focus();
                packageEl.select();
                if (!document.execCommand('copy')) throw new Error('浏览器拒绝复制');
            }
            alert('复制成功！');
        } catch (err) {
            alert('复制失败: ' + err);
        }
    }
    </script>
    """

    items_html = []
    async with debug_log_lock:
        logs = list(DEBUG_LOGS)
    
    for idx, item in enumerate(logs):
        req_id = item.get("id", "N/A")
        req_model = item.get("request_model", "N/A")
        final_model = item.get("final_model", "N/A")
        key_used = item.get("key", "N/A")
        status_code = item.get("status_code", "N/A")
        time_cost = item.get("time_cost", "N/A")
        log_time = item.get("time", "")
        
        raw_req_body = str(item.get("request_body", ""))
        raw_resp_body = str(item.get("response_body", ""))
        req_body = html_lib.escape(raw_req_body)
        resp_body = html_lib.escape(raw_resp_body)
        debug_package = f"""[{log_time}] ID: {req_id} | Status: {status_code} | Time: {time_cost}s
• 请求模型: {req_model}

• 最终模型: {final_model} (Key: ****{key_used})

【Request Body】:
{raw_req_body}

【Response Body】:
{raw_resp_body}"""
        escaped_package = html_lib.escape(debug_package)

        card = f"""
        <div style="background:#111827; border:1px solid #374151; border-radius:8px; padding:15px; margin-bottom:15px;">
            <details>
                <summary style="cursor:pointer; color:#60a5fa; font-weight:bold;">
                    #{idx+1} | [{log_time}] ID: {req_id} | Status: {status_code} | Time: {time_cost}s
                </summary>
                <div style="margin-top:10px; color:#9ca3af; font-size:13px;">
                    <p style="margin:4px 0;">• <b>请求模型:</b> {req_model}</p>
                    <p style="margin:4px 0;">• <b>最终模型:</b> {final_model} (Key: ****{key_used})</p>
                </div>
                <div style="margin-top:10px;">
                    <strong style="color:#60a5fa;">【Request Body】:</strong>
                    <pre id="req-body-{idx}" style="background:#1f2937; padding:10px; border-radius:4px; overflow-x:auto; max-height:300px; color:#f3f4f6; font-size:12px;">{req_body}</pre>
                </div>
                <div style="margin-top:10px;">
                    <strong style="color:#34d399;">【Response Body】:</strong>
                    <pre id="resp-body-{idx}" style="background:#1f2937; padding:10px; border-radius:4px; overflow-x:auto; max-height:300px; color:#f3f4f6; font-size:12px;">{resp_body}</pre>
                    <textarea id="debug-package-{idx}" readonly aria-hidden="true" style="position:fixed; left:-9999px; top:0; width:1px; height:1px;">{escaped_package}</textarea>
                    <button type="button" onclick="copyDebugInfo('{idx}')" style="margin-top:8px; padding:6px 12px; background:#3b82f6; color:#fff; border:none; border-radius:4px; cursor:pointer; font-size:12px;">📋 一键复制 AI 调试包</button>
                </div>
            </details>
        </div>
        """
        items_html.append(card)

    content = "".join(items_html) if items_html else "<p style='color:#9ca3af;'>暂无 Debug 日志记录</p>"
    body_html = f"""
    {copy_script}
    <h2>🔍 全量 Debug 深度调试面板 <span class="tag">(最近 50 条)</span></h2>
    <hr style="border-color:#1e293b; margin-bottom:15px;"/>
    {content}
    """
    return HTMLResponse(content=html_page("Debug 深度调试", body_html))

@app.get("/config", response_class=HTMLResponse)
async def config(request: Request):
    if not is_admin_authenticated(request):
        return admin_required_response()
    agnes_counts = get_agnes_counts()
    body = f"""<h2>⚙️ 系统当前全局运行配置</h2>
<pre style='background:#1f2937; color:#f3f4f6; padding:15px; border-radius:6px; border:1px solid #374151;'>
网关核心版本: {VERSION}
开源 GitHub: https://github.com/xyrct301/cerebras-proxy-re
全局默认模型: {DEFAULT_MODEL}
Cerebras Key 总计: {len(CEREBRAS_API_KEYS)} 个
Groq Key 总计: {len(groq_pool.keys)} 个
Agnes 中国站 Key: {agnes_counts['keys_by_site'].get('cn', 0)} 个
Agnes 国际站 Key: {agnes_counts['keys_by_site'].get('intl', 0)} 个
思考强控制模式 (Thinking): {THINKING_MODE.upper()}
模型自动降级模式 (Fallback): {MODEL_FALLBACK_MODE.upper()}
Upstash 持久化状态: {'🟢 已启用' if UPSTASH_REDIS_REST_URL else '⚪ 本地JSON模式'}
</pre>"""
    return HTMLResponse(content=html_page("Config 配置", body))

@app.api_route("/admin", methods=["GET", "POST"], response_class=HTMLResponse)
async def admin_page(request: Request):
    if not ADMIN_API_KEY:
        return HTMLResponse(content=html_page("Admin", "<h2>Admin 未启用</h2><p>请先设置 <code>ADMIN_API_KEY</code>。</p>"), status_code=503)

    message = ""
    generated_key = ""
    form = await request.form() if request.method == "POST" else {}

    if request.method == "POST" and form.get("action") == "login":
        client_ip = request.client.host if request.client else "unknown"
        admitted = await admit_fixed_window(
            async_client, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN,
            "admin", client_ip, "login", 10,
        )
        if admitted is False:
            return HTMLResponse(content=html_page("Admin", "<h2>登录尝试过多</h2><p>请稍后再试。</p>"), status_code=429)
        if admitted is None:
            return HTMLResponse(content=html_page("Admin", "<h2>登录暂不可用</h2><p>无法校验登录限额。</p>"), status_code=503)
        supplied = str(form.get("admin_key", ""))
        if hmac.compare_digest(supplied, ADMIN_API_KEY):
            response = HTMLResponse(content=html_page("Admin", "<h2>登录成功</h2><p><a href='/admin'>进入管理页面</a></p>"))
            response.set_cookie(
                "admin_session", create_admin_session_token(), max_age=86400,
                httponly=True, secure=request.url.scheme == "https", samesite="strict",
            )
            return response
        message = "管理员密钥错误"

    if not is_admin_authenticated(request):
        login = f"""
        <h2>🔐 Gateway Admin</h2>
        <p style="color:#f87171;">{html_lib.escape(message)}</p>
        <form method="POST">
            <input type="hidden" name="action" value="login"/>
            <input type="password" name="admin_key" placeholder="ADMIN_API_KEY" required style="width:min(360px,100%); box-sizing:border-box; padding:10px; background:#1f2937; color:#fff; border:1px solid #374151; border-radius:4px;"/>
            <button type="submit" style="padding:10px 16px; background:#2563eb; color:#fff; border:0; border-radius:4px;">登录</button>
        </form>"""
        return HTMLResponse(content=html_page("Admin 登录", login), status_code=401 if message else 200)

    if request.method == "POST" and form.get("action") != "login":
        csrf = str(form.get("csrf", ""))
        if not hmac.compare_digest(csrf, admin_csrf_token(request)):
            return HTMLResponse(content=html_page("Admin", "<h2>请求已拒绝</h2><p>CSRF 校验失败。</p>"), status_code=403)
        if form.get("action") == "logout":
            response = HTMLResponse(content=html_page("Admin", "<h2>已退出</h2><p><a href='/admin'>重新登录</a></p>"))
            response.delete_cookie("admin_session")
            return response
        if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
            message = "未配置 Upstash，Admin 当前为只读模式"
        else:
            action = str(form.get("action", ""))
            if action == "create":
                allowed_models = [m.strip() for m in str(form.get("allowed_models", "")).split(",") if m.strip()]
                try:
                    providers = {
                        "cerebras": max(0, int(form.get("cerebras_contributions", 0))),
                        "groq": max(0, int(form.get("groq_contributions", 0))),
                        "agnes": max(0, int(form.get("agnes_contributions", 0))),
                    }
                except (TypeError, ValueError):
                    providers = {"cerebras": 0, "groq": 0, "agnes": 0}
                created = None
                if any(value > 0 for value in providers.values()):
                    created = await access_manager.create({
                        "name": str(form.get("name", "")),
                        "providers": providers,
                        "allowed_models": allowed_models or None,
                        "enabled": True,
                    }, upstash_set_async)
                else:
                    message = "至少需要为一个服务商设置贡献数量"
                if created:
                    generated_key = created["key"]
                    message = "Key 创建成功，仅在本页显示一次"
                elif not message:
                    message = "Key 创建失败"
            elif action == "toggle":
                ok = await access_manager.update_by_client_id(
                    str(form.get("client_id", "")),
                    {"enabled": str(form.get("enabled", "false")).lower() == "true"},
                    upstash_set_async,
                )
                message = "状态已更新" if ok else "状态更新失败"
            elif action == "update":
                try:
                    providers = {
                        "cerebras": max(0, int(form.get("cerebras_contributions", 0))),
                        "groq": max(0, int(form.get("groq_contributions", 0))),
                        "agnes": max(0, int(form.get("agnes_contributions", 0))),
                    }
                except (TypeError, ValueError):
                    providers = {"cerebras": 0, "groq": 0, "agnes": 0}
                allowed_models = [m.strip() for m in str(form.get("allowed_models", "")).split(",") if m.strip()]
                ok = False
                if any(value > 0 for value in providers.values()):
                    ok = await access_manager.update_by_client_id(
                        str(form.get("client_id", "")),
                        {"name": str(form.get("name", "")), "providers": providers, "allowed_models": allowed_models or None},
                        upstash_set_async,
                    )
                message = "配置已更新" if ok else "配置更新失败，至少需要启用一个服务商"
            elif action == "delete":
                ok = await access_manager.delete_by_client_id(str(form.get("client_id", "")), upstash_set_async)
                message = "Key 已删除或禁用" if ok else "删除失败"

    clients = await access_manager.list_clients()
    rows = []
    csrf = admin_csrf_token(request)
    for item in clients:
        models = ", ".join(item["allowed_models"] or ["全部允许范围内模型"])
        providers = item["providers"]
        form_id = f"edit-{item['client_id']}"
        next_enabled = "false" if item["enabled"] else "true"
        rows.append(f"""
        <tr>
            <td><input form="{form_id}" name="name" value="{html_lib.escape(item['name'] or '')}" style="width:120px;"/></td>
            <td style="white-space:nowrap;">C <input form="{form_id}" name="cerebras_contributions" type="number" min="0" max="100" value="{providers.get('cerebras', 0)}" style="width:48px;"/> G <input form="{form_id}" name="groq_contributions" type="number" min="0" max="100" value="{providers.get('groq', 0)}" style="width:48px;"/> A <input form="{form_id}" name="agnes_contributions" type="number" min="0" max="100" value="{providers.get('agnes', 0)}" style="width:48px;"/></td>
            <td>****{html_lib.escape(item['key_suffix'])}</td>
            <td><input form="{form_id}" name="allowed_models" value="{html_lib.escape(', '.join(item['allowed_models'] or []))}" placeholder="留空表示全部" style="min-width:240px;"/></td><td>{'启用' if item['enabled'] else '禁用'}</td>
            <td style="white-space:nowrap;">
                <form id="{form_id}" method="POST"><input type="hidden" name="csrf" value="{csrf}"/><input type="hidden" name="action" value="update"/><input type="hidden" name="client_id" value="{item['client_id']}"/></form><button type="submit" form="{form_id}">保存</button>
                <form method="POST" style="display:inline;"><input type="hidden" name="csrf" value="{csrf}"/><input type="hidden" name="action" value="toggle"/><input type="hidden" name="client_id" value="{item['client_id']}"/><input type="hidden" name="enabled" value="{next_enabled}"/><button type="submit">{'禁用' if item['enabled'] else '启用'}</button></form>
                <form method="POST" style="display:inline;"><input type="hidden" name="csrf" value="{csrf}"/><input type="hidden" name="action" value="delete"/><input type="hidden" name="client_id" value="{item['client_id']}"/><button type="submit">删除</button></form>
            </td>
        </tr>""")

    generated_html = f"<pre style='background:#065f46;padding:12px;'>{html_lib.escape(generated_key)}</pre>" if generated_key else ""
    readonly = "<p style='color:#f59e0b;'>未配置 Upstash，动态配置不可写。</p>" if not UPSTASH_REDIS_REST_URL else ""
    body = f"""
    <h2>🔑 客户端 Key 管理</h2>
    <form method="POST" style="float:right;"><input type="hidden" name="csrf" value="{csrf}"/><button name="action" value="logout" type="submit">退出登录</button></form>
    <p>{html_lib.escape(message)}</p>{generated_html}{readonly}
    <h3>创建 Key</h3>
    <form method="POST" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;">
        <input type="hidden" name="csrf" value="{csrf}"/><input type="hidden" name="action" value="create"/>
        <input name="name" placeholder="名称/备注" required/>
        <input name="cerebras_contributions" type="number" min="0" max="100" value="0" placeholder="Cerebras 贡献 Key 数" required/>
        <input name="groq_contributions" type="number" min="0" max="100" value="0" placeholder="Groq 贡献 Key 数" required/>
        <input name="agnes_contributions" type="number" min="0" max="100" value="0" placeholder="Agnes 贡献池数" required/>
        <input name="allowed_models" placeholder="允许模型，逗号分隔；留空表示范围内全部"/>
        <button type="submit">生成 Key</button>
    </form>
    <h3 style="margin-top:20px;">现有 Key</h3>
    <div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;">
    <tr><th>名称</th><th>服务商贡献</th><th>Key</th><th>允许模型</th><th>状态</th><th>操作</th></tr>
    {''.join(rows) if rows else '<tr><td colspan="6">暂无 Key</td></tr>'}
    </table></div>"""
    return HTMLResponse(content=html_page("Admin", body))

@app.api_route("/thinkingdisplay", methods=["GET", "POST"], response_class=HTMLResponse)
async def thinkingdisplay_page(request: Request):
    global THINKING_MODE
    if not is_admin_authenticated(request):
        return admin_required_response()
    await refresh_runtime_config_async()
    saved = None
    if request.method == "POST":
        form = await request.form()
        if not hmac.compare_digest(str(form.get("csrf", "")), admin_csrf_token(request)):
            return HTMLResponse(content=html_page("Thinking Control", "<h2>CSRF 校验失败</h2>"), status_code=403)
        mode = str(form.get("mode", ""))
        if mode in ["auto", "on", "off"]:
            if UPSTASH_REDIS_REST_URL:
                saved = await upstash_set_async(RUNTIME_CONFIG_KEY, {
                    "thinking_mode": mode,
                    "fallback_mode": MODEL_FALLBACK_MODE,
                })
                if saved:
                    THINKING_MODE = mode
            else:
                THINKING_MODE = mode
                saved = False
    
    body = f"""<h2>🎯 思考输出渲染控制面</h2>
<p>当前强制策略状态: <strong style="color:#3b82f6;">{THINKING_MODE.upper()}</strong></p>
<p class="tag">{'设置已保存到 Upstash，可跨实例保持。' if saved else ('未配置 Upstash，设置仅当前实例有效。' if not UPSTASH_REDIS_REST_URL else ('设置保存失败。' if saved is False else '设置由 Upstash 跨实例同步。'))}</p>
<hr style="border-color:#1e293b;"/>
<form method="POST" style="display:grid;gap:10px;">
    <input type="hidden" name="csrf" value="{admin_csrf_token(request)}"/>
    <button name="mode" value="auto" type="submit">切换至 AUTO</button>
    <button name="mode" value="on" type="submit">切换至 ON</button>
    <button name="mode" value="off" type="submit">切换至 OFF</button>
</form>"""
    return HTMLResponse(content=html_page("Thinking Control", body))

@app.api_route("/fallbackmode", methods=["GET", "POST"], response_class=HTMLResponse)
async def fallbackmode_page(request: Request):
    global MODEL_FALLBACK_MODE
    if not is_admin_authenticated(request):
        return admin_required_response()
    await refresh_runtime_config_async()
    saved = None
    if request.method == "POST":
        form = await request.form()
        if not hmac.compare_digest(str(form.get("csrf", "")), admin_csrf_token(request)):
            return HTMLResponse(content=html_page("Fallback Control", "<h2>CSRF 校验失败</h2>"), status_code=403)
        mode = str(form.get("mode", ""))
        if mode in ["auto", "off", "force_gpt"]:
            if UPSTASH_REDIS_REST_URL:
                saved = await upstash_set_async(RUNTIME_CONFIG_KEY, {
                    "thinking_mode": THINKING_MODE,
                    "fallback_mode": mode,
                })
                if saved:
                    MODEL_FALLBACK_MODE = mode
            else:
                MODEL_FALLBACK_MODE = mode
                saved = False
    
    body = f"""<h2>🔀 模型降级策略控制面</h2>
<p>当前策略状态: <strong style="color:#eab308;">{MODEL_FALLBACK_MODE.upper()}</strong></p>
<p class="tag">{'设置已保存到 Upstash，可跨实例保持。' if saved else ('未配置 Upstash，设置仅当前实例有效。' if not UPSTASH_REDIS_REST_URL else ('设置保存失败。' if saved is False else '设置由 Upstash 跨实例同步。'))}</p>
<hr style="border-color:#1e293b;"/>
<form method="POST" style="display:grid;gap:10px;">
    <input type="hidden" name="csrf" value="{admin_csrf_token(request)}"/>
    <button name="mode" value="auto" type="submit">切换至 AUTO</button>
    <button name="mode" value="off" type="submit">切换至 OFF</button>
    <button name="mode" value="force_gpt" type="submit">切换至 FORCE_GPT</button>
</form>"""
    return HTMLResponse(content=html_page("Fallback Control", body))

@app.get("/v1/models")
async def models(request: Request):
    principal = await authenticate_request(request)
    if principal is None:
        return api_error(401, "Unauthorized", "auth_error", "unauthorized")
    models_list = [
        {"id": spec.public_id, "object": "model", "created": 1700000000, "owned_by": spec.provider}
        for spec in MODEL_CATALOG.values()
    ]
    models_list = [item for item in models_list if access_manager.authorize(principal, item["id"])]
    return JSONResponse(content={"object": "list", "data": models_list})

@app.get("/health")
async def health():
    agnes_counts = get_agnes_counts()
    return JSONResponse(content={
        "status": "ok", 
        "version": VERSION, 
        "cerebras_keys": len(CEREBRAS_API_KEYS),
        "groq_keys": len(groq_pool.keys),
        "agnes_cn_keys": agnes_counts["keys_by_site"].get("cn", 0),
        "agnes_intl_keys": agnes_counts["keys_by_site"].get("intl", 0)
    })
