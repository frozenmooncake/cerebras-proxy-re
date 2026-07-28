import os
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

# 本地自动加载 .env 环境变量
load_dotenv()

# ======================================================
# VERSION & CHANGELOG
# ======================================================
VERSION = "2.0.3-OpenCode-Strict-Compatible"

# ======================================================
# CONFIG & MODES
# ======================================================
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
DEFAULT_MODEL = "gpt-oss-120b"
GLM_MODEL = "zai-glm-4.7"
GPT_MODEL = "gpt-oss-120b"
KEY_COOLDOWN = 60

THINKING_MODE = os.getenv("THINKING_MODE", "auto").lower()
MODEL_FALLBACK_MODE = os.getenv("MODEL_FALLBACK_MODE", "auto").lower()

STATS_FILE = "/tmp/gateway_stats.json" if os.path.exists("/tmp") else "gateway_stats.json"
POOL_FILE = "/tmp/gateway_pool.json" if os.path.exists("/tmp") else "gateway_pool.json"

DEBUG_MAX_TEXT_LEN = 20000

UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()

# ======================================================
# ENV & AUTH
# ======================================================
CUSTOM_API_KEYS = set(filter(None, os.getenv("CUSTOM_API_KEYS", "").split(",")))
CEREBRAS_API_KEYS = list(filter(None, os.getenv("CEREBRAS_API_KEYS", "").split(",")))

if not CEREBRAS_API_KEYS:
    raise Exception("No CEREBRAS_API_KEYS found in environment variables.")

VALID_DEBUG_PASSWORDS = {k + k for k in CUSTOM_API_KEYS} if CUSTOM_API_KEYS else set()

# ======================================================
# 全局 HTTP 异步客户端
# ======================================================
async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0, connect=10.0),
    limits=httpx.Limits(max_keepalive_connections=50, max_connections=200, keepalive_expiry=60.0),
    http2=True
)

# ======================================================
# UPSTASH REDIS ASYNC HELPER
# ======================================================
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

# ======================================================
# STATISTICS & LOGS
# ======================================================
stats_lock = asyncio.Lock()

def get_default_stats():
    return {
        "total_requests": 0,       
        "fallback_count": 0,       
        "429_count": 0,            
        "truncated_count": 0,      
        "other_models_count": 0,    
        "models": {
            GLM_MODEL: {"requests": 0, "input_tokens": 0, "output_tokens": 0, "tokens": 0},
            GPT_MODEL: {"requests": 0, "input_tokens": 0, "output_tokens": 0, "tokens": 0}
        }
    }

GLOBAL_STATS = get_default_stats()

async def init_global_stats():
    global GLOBAL_STATS
    up_stats = await upstash_get_async("gateway_stats")
    if up_stats and "models" in up_stats and GLM_MODEL in up_stats["models"]:
        GLOBAL_STATS = up_stats
        return
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if "models" in saved and GLM_MODEL in saved["models"]:
                    GLOBAL_STATS = saved
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

async def add_log_async(data: dict):
    async with log_lock:
        REQUEST_LOGS.appendleft(data)

DEBUG_LOGS = deque(maxlen=50)
debug_log_lock = asyncio.Lock()

async def add_debug_log_async(data: dict):
    async with debug_log_lock:
        DEBUG_LOGS.appendleft(data)

def truncate_text(text: str, max_len: int = DEBUG_MAX_TEXT_LEN) -> str:
    if isinstance(text, str) and len(text) > max_len:
        return text[:max_len] + f"\n... [已自动截断，原长度: {len(text)} 字符]"
    return text

# ======================================================
# TIKTOKEN ESTIMATION HELPER
# ======================================================
def estimate_tokens(messages: list = None, text_content: str = "") -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = tiktoken.get_encoding("gpt-4")
    
    if messages:
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            if isinstance(message, dict):
                for key, value in message.items():
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value))
        num_tokens += 2
        return num_tokens
    if text_content:
        return len(encoding.encode(text_content))
    return 0

# ======================================================
# KEY POOL & RATE LIMIT SYNC
# ======================================================
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
            for model in [GLM_MODEL, GPT_MODEL]:
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
                    "limit_rpm": 60,
                    "limit_rpd": 1000,
                    "limit_tpm": 6000000,
                    "limit_tpd": 6000000,
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
                    export[k] = {}
                    for m, info in v.items():
                        export[k][m] = {
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
                if info and info["cooldown"] <= now:
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
        if not key or model not in [GLM_MODEL, GPT_MODEL]:
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
    yield

app = FastAPI(title="Cerebras OpenAI Gateway", version=VERSION, lifespan=lifespan)

# 允许全量的 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# HELPER & AUTH
# ======================================================
def sse(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, ensure_ascii=False)
    return f"data: {data}\n\n"

def check_auth(request: Request) -> bool:
    if not CUSTOM_API_KEYS:
        return True
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    return auth.replace("Bearer ", "").strip() in CUSTOM_API_KEYS

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
    if not usage or model not in GLOBAL_STATS["models"]:
        return
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", prompt + completion)

    async with stats_lock:
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

# ======================================================
# API ENDPOINTS (针对 OpenCode 完美适配)
# ======================================================

# 1. 适配 OpenCode 的 Base URL 测试响应
@app.get("/")
@app.get("/v1")
async def root_v1():
    return JSONResponse(content={"status": "ok", "message": "Cerebras OpenAI Gateway active", "version": VERSION})

# 2. OpenCode 查询可支持模型的全量数据结构
@app.get("/v1/models")
async def models():
    created_ts = int(time.time())
    return JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": GLM_MODEL, 
                "object": "model", 
                "created": created_ts, 
                "owned_by": "cerebras",
                "permission": [],
                "root": GLM_MODEL,
                "parent": None
            }, 
            {
                "id": GPT_MODEL, 
                "object": "model", 
                "created": created_ts, 
                "owned_by": "cerebras",
                "permission": [],
                "root": GPT_MODEL,
                "parent": None
            }
        ]
    })

# 3. 适配 OpenCode 单模型信息获取 (如 GET /v1/models/gpt-oss-120b)
@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    created_ts = int(time.time())
    return JSONResponse(content={
        "id": model_id,
        "object": "model",
        "created": created_ts,
        "owned_by": "cerebras"
    })

# 4. Chat Completions 主调逻辑
@app.post("/v1/chat/completions")
async def chat(request: Request):
    start = time.time()
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if not check_auth(request):
        return JSONResponse(status_code=401, content={
            "error": {"message": "Unauthorized API Key", "type": "auth_error", "param": None, "code": "unauthorized"}
        })

    try:
        raw = await request.json()
    except Exception:
        raw = {}

    async with stats_lock:
        GLOBAL_STATS["total_requests"] += 1
    await save_global_stats_async()

    request_model = raw.get("model", DEFAULT_MODEL)
    
    # 非托管模型透传
    if request_model not in [GLM_MODEL, GPT_MODEL]:
        async with stats_lock:
            GLOBAL_STATS["other_models_count"] += 1
        await save_global_stats_async()
        
        if not CEREBRAS_API_KEYS:
            return JSONResponse(status_code=500, content={"error": {"message": "No physical keys configured"}})
        try:
            r = await async_client.post(
                f"{CEREBRAS_BASE_URL}/chat/completions",
                json=raw,
                headers={"Authorization": f"Bearer {CEREBRAS_API_KEYS[0]}"},
                timeout=60.0
            )
            return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": {"message": str(e)}})

    show_thinking = final_thinking(raw)
    
    models_to_try = []
    if MODEL_FALLBACK_MODE == "force_gpt":
        models_to_try = [GPT_MODEL]
    elif MODEL_FALLBACK_MODE == "off":
        models_to_try = [request_model]
    else:  # auto
        models_to_try = [request_model]
        if request_model == GLM_MODEL:
            models_to_try.append(GPT_MODEL)
    
    last_error = None
    final_model_used = None
    fallback_happened = False
    
    for target_model in models_to_try:
        if target_model != request_model:
            fallback_happened = True
        
        body = sanitize_body(raw, show_thinking, target_model=target_model)
        tried_keys = set()
        max_attempts = len(CEREBRAS_API_KEYS)
        
        for attempt in range(max_attempts):
            selected_key, selected_model = await pool.get_next_key_for_request(target_model, tried_keys)
            if not selected_key:
                break
            
            tried_keys.add(selected_key)
            estimated_in = estimate_tokens(messages=raw.get("messages", []))
            
            # --- STREAM MODE ---
            if body["stream"]:
                body["model"] = selected_model
                await pool.record_request_attempt_async(selected_key, selected_model, estimated_in)

                async def event_generator():
                    nonlocal last_error, final_model_used
                    last_usage = None
                    generated_text = ""
                    stream_chunks = []
                    is_truncated = False
                    response = None
                    created_time = int(time.time())

                    try:
                        req = async_client.build_request(
                            "POST",
                            f"{CEREBRAS_BASE_URL}/chat/completions",
                            json=body,
                            headers={"Authorization": f"Bearer {selected_key}", "Content-Type": "application/json"},
                            timeout=60.0
                        )
                        response = await async_client.send(req, stream=True)

                        if response.status_code == 200:
                            await pool.sync_headers_async(selected_key, selected_model, response.headers)
                            await pool.success_async(selected_key, selected_model)
                            final_model_used = selected_model

                            # 关键：先发送一个标准的 role 初始帧，确保 OpenCode 建立流上下文
                            first_chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": selected_model,
                                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
                            }
                            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

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

                                    # 规范性对齐
                                    obj["id"] = request_id
                                    obj["object"] = "chat.completion.chunk"
                                    obj["created"] = created_time
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
                                    yield chunk_out
                                except Exception:
                                    continue

                        elif response.status_code == 429:
                            async with stats_lock: GLOBAL_STATS["429_count"] += 1
                            await save_global_stats_async()
                            await pool.cooldown_async(selected_key, selected_model)
                            await pool.sync_headers_async(selected_key, selected_model, response.headers)
                            err_obj = {"error": {"message": f"429 Limit reached", "type": "requests_limit", "code": 429}}
                            yield sse(err_obj)
                            return
                        else:
                            err_obj = {"error": {"message": f"Upstream error {response.status_code}", "type": "api_error", "code": response.status_code}}
                            yield sse(err_obj)
                            return

                    except Exception as e:
                        await pool.cooldown_async(selected_key, selected_model)
                        err_obj = {"error": {"message": str(e), "type": "internal_error", "code": 500}}
                        yield sse(err_obj)
                        return

                    finally:
                        if response and response.status_code == 200:
                            if is_truncated:
                                async with stats_lock: GLOBAL_STATS["truncated_count"] += 1
                                await save_global_stats_async()

                            if not last_usage:
                                out_tokens = estimate_tokens(text_content=generated_text)
                                last_usage = {"prompt_tokens": estimated_in, "completion_tokens": out_tokens, "total_tokens": estimated_in + out_tokens}

                            await add_global_usage_async(selected_model, last_usage)
                            await pool.sync_headers_async(selected_key, selected_model, response.headers, actual_tokens=last_usage.get("total_tokens", 0))
                            
                            if fallback_happened:
                                async with stats_lock: GLOBAL_STATS["fallback_count"] += 1
                                await save_global_stats_async()
                            
                            await finish_log_async(request_id, request_model, selected_key, selected_model, fallback_happened, start, last_usage)

                            if response:
                                await response.aclose()

                return StreamingResponse(
                    event_generator(), 
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"  # 防止 Nginx 缓存导致流阻塞
                    }
                )

            # --- NON-STREAM MODE ---
            body["model"] = selected_model
            await pool.record_request_attempt_async(selected_key, selected_model, estimated_in)
            
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
                    final_model_used = selected_model
                    
                    usage = result.get("usage")
                    if not usage:
                        text_out = ""
                        for choice in result.get("choices", []):
                            msg = choice.get("message", {})
                            if "content" in msg and msg["content"]: text_out += msg["content"]
                        out_tokens = estimate_tokens(text_content=text_out)
                        usage = {"prompt_tokens": estimated_in, "completion_tokens": out_tokens, "total_tokens": estimated_in + out_tokens}
                        result["usage"] = usage

                    await add_global_usage_async(selected_model, usage)
                    await pool.sync_headers_async(selected_key, selected_model, r.headers, actual_tokens=usage.get("total_tokens", 0))
                    
                    if not show_thinking:
                        result = remove_thinking(result)

                    if fallback_happened:
                        async with stats_lock: GLOBAL_STATS["fallback_count"] += 1
                        await save_global_stats_async()

                    await finish_log_async(request_id, request_model, selected_key, selected_model, fallback_happened, start, usage)

                    return JSONResponse(status_code=200, content=result)
                
                elif r.status_code == 429:
                    async with stats_lock: GLOBAL_STATS["429_count"] += 1
                    await save_global_stats_async()
                    await pool.cooldown_async(selected_key, selected_model)
                    await pool.sync_headers_async(selected_key, selected_model, r.headers)
                    last_error = {"type": "429", "key": selected_key[-4:], "model": selected_model}
                    continue
                
                else:
                    last_error = {"type": "http_error", "code": r.status_code, "key": selected_key[-4:], "model": selected_model}
                    continue
                    
            except Exception as e:
                await pool.cooldown_async(selected_key, selected_model)
                last_error = {"type": "exception", "message": str(e), "key": selected_key[-4:], "model": selected_model}
                continue
        
    error_msg = f"All keys/models failed. Last error: {json.dumps(last_error)}" if last_error else "No keys available"
    return JSONResponse(status_code=503, content={
        "error": {"message": error_msg, "type": "service_unavailable", "param": None, "code": "all_keys_failed"}
    })

# 保留轻量 HTML 管理界面
@app.get("/menu", response_class=HTMLResponse)
@app.get("/status", response_class=HTMLResponse)
@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok", "version": VERSION, "keys_loaded": len(CEREBRAS_API_KEYS)})