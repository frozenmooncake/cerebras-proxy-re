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

# 本地自动加载 .env 环境变量（解决 Windows CMD 终端变量不继承问题）
load_dotenv()

# ======================================================
# VERSION & CHANGELOG
# ======================================================
VERSION = "2.0.3"

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

# Serverless 临时目录兼容
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
# 全局 HTTP 异步客户端 (网络与长连接性能优化)
# ======================================================
async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0, connect=10.0),
    limits=httpx.Limits(max_keepalive_connections=50, max_connections=200, keepalive_expiry=60.0),
    http2=True  # 启用 HTTP/2 复用 TCP 连接，显著降低首字延迟
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
# STATISTICS & LOGS (全异步锁)
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

# FastAPI Lifespan 结构
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_global_stats()
    await pool.init_pool_data()
    yield

app = FastAPI(title="Cerebras OpenAI Gateway", version=VERSION, lifespan=lifespan)

# 增加 CORS 全局中间件兼容 OpenCode/Web客户端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# HTML HELPER & AUTH
# ======================================================
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
</style>
</head>
<body>
<div class="box">{body}</div>
<div class="nav"><a href="/menu" class="nav-btn">🔙 返回主菜单</a></div>
</body>
</html>"""

def check_auth(request: Request) -> bool:
    if not CUSTOM_API_KEYS:
        return True
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    return auth.replace("Bearer ", "").strip() in CUSTOM_API_KEYS

# ======================================================
# THINKING & BODY SANITIZE
# ======================================================
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
# API ENDPOINT: CHAT COMPLETIONS (Async Streaming)
# ======================================================
@app.post("/v1/chat/completions")
async def chat(request: Request):
    start = time.time()
    request_id = str(uuid.uuid4())[:8]

    if not check_auth(request):
        return JSONResponse(status_code=401, content={
            "error": {"message": "Unauthorized", "type": "auth_error", "param": None, "code": "unauthorized"},
            "request_id": request_id
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
            bj_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
            await add_debug_log_async({
                "id": request_id, "time": bj_time, "request_model": request_model,
                "final_model": request_model, "key": CEREBRAS_API_KEYS[0][-4:], "status_code": r.status_code,
                "time_cost": round(time.time() - start, 2),
                "request_body": truncate_text(json.dumps(raw, ensure_ascii=False, indent=2)),
                "response_body": truncate_text(r.text)
            })
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

                                    # 修正：补充兼容 OpenCode 等编辑器解析的标准 SSE 字段
                                    if "id" not in obj: obj["id"] = f"chatcmpl-{request_id}"
                                    if "object" not in obj: obj["object"] = "chat.completion.chunk"
                                    if "created" not in obj: obj["created"] = int(time.time())
                                    if "model" not in obj: obj["model"] = selected_model

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

                        elif response.status_code == 429:
                            async with stats_lock: GLOBAL_STATS["429_count"] += 1
                            await save_global_stats_async()
                            await pool.cooldown_async(selected_key, selected_model)
                            await pool.sync_headers_async(selected_key, selected_model, response.headers)
                            err_obj = {"error": {"message": f"429 Limit reached on ****{selected_key[-4:]}", "type": "requests_limit", "code": 429}}
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

                            bj_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
                            trunc_flag = " [⚠️ 长度截断]" if is_truncated else ""
                            resp_debug_str = f"【流式输出 (Stream Text)】{trunc_flag}:\n{generated_text}\n\n【原始 SSE Chunk 样例】:\n" + "".join(stream_chunks[:10])
                            await add_debug_log_async({
                                "id": request_id, "time": bj_time, "request_model": request_model,
                                "final_model": selected_model, "key": selected_key[-4:], "status_code": 200,
                                "time_cost": round(time.time() - start, 2),
                                "request_body": truncate_text(json.dumps(raw, ensure_ascii=False, indent=2)),
                                "response_body": truncate_text(resp_debug_str)
                            })
                            if response:
                                await response.aclose()

                return StreamingResponse(event_generator(), media_type="text/event-stream")

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
                    is_truncated = False
                    
                    for choice in result.get("choices", []):
                        if choice.get("finish_reason") == "length":
                            is_truncated = True

                    if is_truncated:
                        async with stats_lock: GLOBAL_STATS["truncated_count"] += 1
                        await save_global_stats_async()

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

                    bj_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
                    if is_truncated: result["_gateway_warning"] = "⚠️ 触发长度截断 (finish_reason: length)"
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
    bj_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    await add_debug_log_async({
        "id": request_id, "time": bj_time, "request_model": request_model,
        "final_model": "None", "key": "None", "status_code": 503,
        "time_cost": round(time.time() - start, 2),
        "request_body": truncate_text(json.dumps(raw, ensure_ascii=False, indent=2)),
        "response_body": truncate_text(error_msg)
    })

    return JSONResponse(status_code=503, content={
        "error": {"message": error_msg, "type": "service_unavailable", "param": None, "code": "all_keys_failed"},
        "request_id": request_id
    })

# ======================================================
# PAGES & CONTROL PANEL
# ======================================================
@app.get("/menu", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def menu():
    body = f"""<h2>🧠 Cerebras OpenAI API Gateway Menu</h2>
<p><strong>Version:</strong> {VERSION}</p>
<p style="color: #9ca3af; font-size: 14px; margin-top: -10px;">作者：速冻月饼</p>
<hr style="border-color:#1e293b;"/>
<h3>📌 API Endpoint</h3>
<p>🔗 <a href="/v1/models" target="_blank">/v1/models (查看可用模型列表)</a></p>
<p> <code>POST /v1/chat/completions</code> (标准 OpenAI 格式网关)</p>
<h3>📊 监控中心 (Monitor)</h3>
<p>📈 <a href="/status">/status (实时上游限额 & 物理Key高级看板)</a></p>
<p>📜 <a href="/log">/log (最近 100 条请求历史)</a></p>
<p>🔍 <a href="/debug">/debug (最近 50 条全量请求体/响应体深度调试)</a></p>
<p>⚙️ <a href="/config">/config (系统核心配置)</a></p>
<p>❤️ <a href="/health">/health (微服务健康检查)</a></p>
<h3>🎛️ 控制策略 (Control Center)</h3>
<p>⚙️ <a href="/thinkingdisplay">/thinkingdisplay (深度思考输出控制: {THINKING_MODE.upper()})</a></p>
<p>🔀 <a href="/fallbackmode">/fallbackmode (GLM->GPT 降级策略控制: {MODEL_FALLBACK_MODE.upper()})</a></p>
<h3> 默认托管模型</h3>
<pre style="background:#1f2937; padding:12px; border-radius:6px; border:1px solid #374151;">{DEFAULT_MODEL}</pre>"""
    return HTMLResponse(content=html_page("Menu", body))

@app.get("/status", response_class=HTMLResponse)
async def status():
    now = time.time()
    global_limits = {
        GLM_MODEL: {"rpm": 0, "rpd": 0, "tpm": 0, "tpd": 0, "cur_rpm": 0, "cur_rpd": 0, "cur_tpm": 0, "cur_tpd": 0},
        GPT_MODEL: {"rpm": 0, "rpd": 0, "tpm": 0, "tpd": 0, "cur_rpm": 0, "cur_rpd": 0, "cur_tpm": 0, "cur_tpd": 0}
    }
    
    for key in CEREBRAS_API_KEYS:
        for model in [GLM_MODEL, GPT_MODEL]:
            metrics = pool.get_current_metrics(key, model)
            global_limits[model]["rpm"] += metrics["limit_rpm"]
            global_limits[model]["rpd"] += metrics["limit_rpd"]
            global_limits[model]["tpm"] += metrics["limit_tpm"]
            global_limits[model]["tpd"] += metrics["limit_tpd"]
            global_limits[model]["cur_rpm"] += metrics["current_rpm"]
            global_limits[model]["cur_rpd"] += metrics["current_rpd"]
            global_limits[model]["cur_tpm"] += metrics["current_tpm"]
            global_limits[model]["cur_tpd"] += metrics["current_tpd"]

    def render_progress(cur, limit):
        pct = min(100, round((cur / limit * 100), 1)) if limit > 0 else 0
        color = "#ef4444" if pct > 80 else ("#eab308" if pct > 50 else "#3b82f6")
        return f"""
        <div class="metric-row"><span>进度: {cur} / {limit}</span><span>{pct}%</span></div>
        <div class="progress-bar-bg"><div class="progress-bar" style="width: {pct}%; background: {color};"></div></div>
        """

    html = f"""
    <h2>📊 Cerebras Gateway 实时监控看板 <span class="tag">v{VERSION}</span></h2>
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
            <div class="tag">🚨 触发长度截断</div>
            <div style="font-size:28px; font-weight:bold; color:#f97316; margin-top:4px;">{GLOBAL_STATS.get('truncated_count', 0)}</div>
        </div>
        <div class="card">
            <div class="tag">上游 429 阻断</div>
            <div style="font-size:28px; font-weight:bold; color:#ef4444; margin-top:4px;">{GLOBAL_STATS['429_count']}</div>
        </div>
        <div class="card">
            <div class="tag">旁路透传模型</div>
            <div style="font-size:28px; font-weight:bold; color:#a855f7; margin-top:4px;">{GLOBAL_STATS['other_models_count']}</div>
        </div>
    </div>
    
    <h3>🔮 托管模型全局汇总 (All Keys Combined)</h3>
    <div class="grid-2">
    """
    
    for model in [GLM_MODEL, GPT_MODEL]:
        g_data = GLOBAL_STATS["models"][model]
        g_lim = global_limits[model]
        html += f"""
        <div class="card">
            <div style="font-size:16px; font-weight:bold; color:#60a5fa; margin-bottom:12px;">📌 模型: {model}</div>
            <div class="metric-row"><span class="tag">累计处理请求:</span> <strong>{g_data['requests']}</strong></div>
            <div class="metric-row"><span class="tag">输入 / 输出 Tokens:</span> <span>{g_data['input_tokens']} / {g_data['output_tokens']}</span></div>
            <div class="metric-row" style="border-bottom:1px solid #374151; padding-bottom:8px; margin-bottom:12px;"><span class="tag">历史总耗费 Tokens:</span> <strong style="color:#10b981;">{g_data['tokens']}</strong></div>
            
            <div style="font-size:13px; font-weight:bold; margin-bottom:6px; color:#9ca3af;">实时全负载动态额度水位:</div>
            <div class="tag">RPM (每分钟请求数)</div>{render_progress(g_lim['cur_rpm'], g_lim['rpm'])}
            <div class="tag">RPD (每日请求数)</div>{render_progress(g_lim['cur_rpd'], g_lim['rpd'])}
            <div class="tag">TPM (每分钟Tokens)</div>{render_progress(g_lim['cur_tpm'], g_lim['tpm'])}
            <div class="tag">TPD (每日Tokens)</div>{render_progress(g_lim['cur_tpd'], g_lim['tpd'])}
        </div>
        """
    html += "</div>"

    html += "<h3>🔑 物理 API-Key 状态细节矩阵</h3><div class='grid-2'>"
    for key in CEREBRAS_API_KEYS:
        k_suffix = key[-4:] if len(key) > 4 else key
        html += f"""
        <div class="card" style="border-left: 4px solid #3b82f6;">
            <div style="font-weight:bold; font-size:15px; margin-bottom:12px;">🗝️ Key: ****{k_suffix}</div>
        """
        for model in [GLM_MODEL, GPT_MODEL]:
            info = pool.data[key][model]
            cd = max(0, int(info["cooldown"] - now))
            status_badge = f'<span class="badge badge-red">🔴 冷却中 ({cd}s)</span>' if cd > 0 else '<span class="badge badge-green">🟢 正常</span>'
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

    return HTMLResponse(content=html_page("Status 看板", html))

@app.get("/log", response_class=HTMLResponse)
async def log_page():
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

@app.api_route("/debug", methods=["GET", "POST"], response_class=HTMLResponse)
async def debug(request: Request, password: Optional[str] = Form(None), debug_auth_token: Optional[str] = Cookie(None)):
    if CUSTOM_API_KEYS:
        auth_token = debug_auth_token or password
        valid_set = CUSTOM_API_KEYS | VALID_DEBUG_PASSWORDS
        if not auth_token or auth_token not in valid_set:
            login_form = """
            <h2>🔒 Debug 面板访问受限</h2>
            <form method="POST">
                <input type="password" name="password" placeholder="请输入 Debug 访问密钥" style="padding:8px; border-radius:4px; border:1px solid #374151; background:#1f2937; color:#fff;"/>
                <button type="submit" style="padding:8px 15px; background:#3b82f6; color:#fff; border:none; border-radius:4px; cursor:pointer;">登录</button>
            </form>
            """
            return HTMLResponse(content=html_page("Debug 鉴权", login_form))

    copy_script = """
    <script>
    function copyDebugInfo(idx) {
        const respEl = document.getElementById('resp-body-' + idx);
        const text = respEl ? respEl.innerText : '';
        navigator.clipboard.writeText(text).then(() => { alert('复制成功！'); }).catch(err => { alert('复制失败: ' + err); });
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
        
        req_body = str(item.get("request_body", "")).replace("<", "&lt;").replace(">", "&gt;")
        resp_body = str(item.get("response_body", "")).replace("<", "&lt;").replace(">", "&gt;")

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
                    <pre style="background:#1f2937; padding:10px; border-radius:4px; overflow-x:auto; max-height:300px; color:#f3f4f6; font-size:12px;">{req_body}</pre>
                </div>
                <div style="margin-top:10px;">
                    <strong style="color:#34d399;">【Response Body】:</strong>
                    <pre id="resp-body-{idx}" style="background:#1f2937; padding:10px; border-radius:4px; overflow-x:auto; max-height:300px; color:#f3f4f6; font-size:12px;">{resp_body}</pre>
                    <button onclick="copyDebugInfo('{idx}')" style="margin-top:8px; padding:6px 12px; background:#3b82f6; color:#fff; border:none; border-radius:4px; cursor:pointer; font-size:12px;">📋 一键复制 AI 调试包</button>
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
    resp = HTMLResponse(content=html_page("Debug 深度调试", body_html))
    if CUSTOM_API_KEYS and password:
        resp.set_cookie(key="debug_auth_token", value=password, max_age=86400*7)
    return resp

@app.get("/config", response_class=HTMLResponse)
async def config():
    body = f"""<h2>⚙️ 系统当前全局运行配置</h2>
<pre style='background:#1f2937; color:#f3f4f6; padding:15px; border-radius:6px; border:1px solid #374151;'>
网关核心版本: {VERSION}
全局默认模型: {DEFAULT_MODEL}
GLM模型映射: {GLM_MODEL}
GPT模型映射: {GPT_MODEL}
API-Key总计: {len(CEREBRAS_API_KEYS)} 个物理Key
触发限流冷却: {KEY_COOLDOWN}秒
思考强控制模式 (Thinking): {THINKING_MODE.upper()}
模型自动降级模式 (Fallback): {MODEL_FALLBACK_MODE.upper()}
Upstash 持久化状态: {'🟢 已启用' if UPSTASH_REDIS_REST_URL else '⚪ 本地JSON模式'}
</pre>"""
    return HTMLResponse(content=html_page("Config 配置", body))

@app.get("/thinkingdisplay", response_class=HTMLResponse)
async def thinkingdisplay_page(mode: Optional[str] = None):
    global THINKING_MODE
    if mode in ["auto", "on", "off"]:
        THINKING_MODE = mode
    
    body = f"""<h2>🎯 思考输出渲染控制面</h2>
<p>当前强制策略状态: <strong style="color:#3b82f6;">{THINKING_MODE.upper()}</strong></p>
<hr style="border-color:#1e293b;"/>
<p>💡 <b>状态说明:</b><br/>
- <b>AUTO:</b> 智能遵循客户端参数（未传默认关闭）。<br/>
- <b>ON:</b> 强制完整保留并输出所有的思考推理字段。<br/>
- <b>OFF:</b> 强制在上游屏蔽并在响应中彻底抹除推理内容（极速 + 节省 Token）。</p>
<br/>
<ul style="list-style:none; padding:0;">
    <li style="margin-bottom:10px;"><a href="/thinkingdisplay?mode=auto" style="display:block; padding:10px; background:#1f2937; border-radius:6px; border:1px solid #374151;">🔄 切换至 AUTO</a></li>
    <li style="margin-bottom:10px;"><a href="/thinkingdisplay?mode=on" style="display:block; padding:10px; background:#065f46; color:#34d399; border-radius:6px;">🟢 切换至 ON</a></li>
    <li style="margin-bottom:10px;"><a href="/thinkingdisplay?mode=off" style="display:block; padding:10px; background:#991b1b; color:#f87171; border-radius:6px;">🔴 切换至 OFF (强烈推荐)</a></li>
</ul>"""
    return HTMLResponse(content=html_page("Thinking Control", body))

@app.get("/fallbackmode", response_class=HTMLResponse)
async def fallbackmode_page(mode: Optional[str] = None):
    global MODEL_FALLBACK_MODE
    if mode in ["auto", "off", "force_gpt"]:
        MODEL_FALLBACK_MODE = mode
    
    body = f"""<h2>🔀 模型降级策略控制面 (GLM -> GPT)</h2>
<p>当前策略状态: <strong style="color:#eab308;">{MODEL_FALLBACK_MODE.upper()}</strong></p>
<hr style="border-color:#1e293b;"/>
<p>💡 <b>状态说明:</b><br/>
- <b>AUTO:</b> 智能自动降级。当 GLM 出现 429/5xx 错误时，无缝切换至 GPT 接管。<br/>
- <b>OFF:</b> 严禁降级。严格只使用 GLM 模型，报错直接返回给客户端。<br/>
- <b>FORCE_GPT:</b> 应急切流。所有请求直接强制切流至 GPT 处理。</p>
<br/>
<ul style="list-style:none; padding:0;">
    <li style="margin-bottom:10px;"><a href="/fallbackmode?mode=auto" style="display:block; padding:10px; background:#1f2937; border-radius:6px; border:1px solid #374151;">🔄 切换至 AUTO (默认)</a></li>
    <li style="margin-bottom:10px;"><a href="/fallbackmode?mode=off" style="display:block; padding:10px; background:#991b1b; color:#f87171; border-radius:6px;">🔴 切换至 OFF (禁止降级)</a></li>
    <li style="margin-bottom:10px;"><a href="/fallbackmode?mode=force_gpt" style="display:block; padding:10px; background:#1e3a8a; color:#60a5fa; border-radius:6px;">⚡ 切换至 FORCE_GPT (强切GPT)</a></li>
</ul>"""
    return HTMLResponse(content=html_page("Fallback Control", body))

# 修正：补全 OpenAI 标准 created / owned_by 字段，解决 OpenCode / Cursor 无法识别模型列表的问题
@app.get("/v1/models")
async def models():
    return JSONResponse(content={
        "object": "list",
        "data": [
            {"id": GLM_MODEL, "object": "model", "created": 1700000000, "owned_by": "cerebras"}, 
            {"id": GPT_MODEL, "object": "model", "created": 1700000000, "owned_by": "cerebras"}
        ]
    })

@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok", "version": VERSION, "keys_loaded": len(CEREBRAS_API_KEYS)})
