# Cerebras OpenAI API Gateway 2.1.3

> 聚合 Cerebras、Groq 和 Agnes 的 OpenAI 兼容网关，支持多 Key 轮询、模型权限和贡献额度管理。

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![HTTP2 Enabled](https://img.shields.io/badge/HTTP%2F2-Enabled-brightgreen)

[功能特性](#-功能特性) · [控制面板](#-可视化控制面板) · [环境变量](#-环境变量配置) · [全平台部署指南](#-全平台部署指南) · [高可用主备架构](#-高可用双活主备方案) · [客户端接入](#-客户端接入说明)

---

## 🚀 功能特性

- **极速流式响应 (Stream Optimization)**：采用 **FastAPI + HTTP/2 Multiplexing + 长连接复用**，大幅降低首字延迟（TTFT），支持毫秒级打字机推流。
- **多物理 Key 智能轮询 (Key Pool)**：支持配置多个 Cerebras API-Key，自动负载均衡、计算限额并在遇到 `429 Too Many Requests` 时自动触发 60s 智能冷却隔离。
- **智能模型自动降级 (Fallback System)**：当上游 GLM 模型 (`zai-glm-4.7`) 发生限流或异常时，可无缝无感自动降级至 GPT 模型 (`gpt-oss-120b`) 接管响应。
- **Groq 兼容降级 (Groq Fallback)**：Cerebras 全部失败后，自动切换到 Groq 托管模型，继续提供 OpenAI 兼容响应。
- **Agnes 双站轮询**：中国站和国际站分别配置物理 Key，按“站点 + Key”候选节点轮询，支持文本、图片和异步视频接口。
- **客户端 Key 权限**：每个 Key 分别配置 Cerebras、Groq、Agnes 的贡献数量，数量为 `0` 的服务商不可访问；还可设置精确模型白名单。
- **贡献额度**：每个客户端 Key、每个服务商独立计算 RPM/TPM，公式为 `官方基准额度 × 对应服务商贡献数量 × 2`。
- **思考过程深度控制 (Thinking Control)**：支持 `AUTO` / `ON` / `OFF` 三档控制，可一键全局强行抹除思考推理过程，极致节省 Token 消耗与响应耗时。
- **可视化监控与深度调试**：提供极简黑夜风格看板，实时查看所有 Key 的 RPM/RPD/TPM/TPD 水位线、最近 100 条请求历史，以及支持同时打包 Request 与 Response 的一键调试包复制。
- **OpenCode / Cursor 兼容**：补齐 OpenAI 标准模型列表与流式字段，便于编辑器直接识别并接入。
- **云端/本地双模持久化**：原生支持 Upstash Redis 异步持久化存储统计与 Key 池状态；无 Redis 时自动平滑回退至本地 JSON 文件存储。
- **日志与控制状态持久化**：配置 Upstash 后，最近 100 条请求日志、最近 50 条 Debug 日志以及 Thinking/Fallback 模式会跨 Vercel 实例同步并在冷启动后恢复。
- **统一模型目录与 Provider Adapter**：模型归属、操作类型和基础额度集中管理；Groq/Agnes 的响应、流式标准化和资源关闭由统一 adapter 负责。
- **分布式上游保护**：配置 Upstash 后，Cerebras、Groq 和 Agnes 的物理上游 RPM admission 在 Vercel 多实例间共享。
- **管理面安全**：状态、日志、Debug、配置和控制页面仅允许 Admin 会话访问；控制修改使用 POST + CSRF，Admin 会话带 24 小时签名有效期。
- **测试基线**：使用标准库 `unittest` 覆盖鉴权、模型目录、共享额度、管理页面、Provider adapter、视频 affinity 和 Redis 限额。
- **图片输入预检**：请求进入上游前校验模型视觉能力和图片地址；客户端本地文件路径返回明确的 `400` 错误，不再表现为流式无回复。

---

## 🎛️ 可视化控制面板

部署成功后，访问网关根路径或相应路由即可进入管理界面：

| 路由页面 | 功能说明 |
| :--- | :--- |
| `/menu` | 网关主菜单与全局配置概览 |
| `/status` | 实时全局负载水位、上游限额看板与物理 Key 冷却状态 |
| `/thinkingdisplay` | 强行控制/抹除 AI 思考推理过程 (`AUTO` / `ON` / `OFF`) |
| `/fallbackmode` | GLM -> GPT 自动降级策略切换 (`AUTO` / `OFF` / `FORCE_GPT`) |
| `/logs` | 最近 100 条请求日志回溯 |
| `/debug` | 最近 50 条全量 Request/Response 抓包与一键打包复制 AI 调试信息 |
| `/admin` | 创建、启用、禁用和删除客户端 Key，配置权限范围、模型白名单和贡献数量 |

### Agnes API 路由

| 路由 | 模型 | 说明 |
| :--- | :--- | :--- |
| `POST /v1/chat/completions` | `agnes/agnes-2.5-flash` | OpenAI 兼容文本与流式响应 |
| `POST /v1/images/generations` | `agnes/agnes-image-2.1-flash` | 图片生成或图片编辑 |
| `POST /v1/videos` | `agnes/agnes-video-v2.0` | 创建异步视频任务 |
| `GET /agnesapi?video_id=...` | `agnes/agnes-video-v2.0` | 查询视频任务结果 |
| `GET /v1/videos/{task_id}` | `agnes/agnes-video-v2.0` | 兼容旧版任务查询 |

---

## ⚙️ 环境变量配置

在部署平台（或本地 `.env` 文件）中配置以下环境变量：

| 环境变量名 | 必填 | 默认值 | 说明 |
| :--- | :---: | :---: | :--- |
| `CEREBRAS_API_KEYS` | **是** | - | 物理 Cerebras API Key，多个用英文逗号分隔，如 `csk-key1,csk-key2` |
| `GROQ_API_KEYS` | 否 | - | Groq API Key，多个用英文逗号分隔，如 `gsk-key1,gsk-key2` |
| `AGNES_CN_API_KEYS` | 否 | - | Agnes 中国站 Key，固定用于 `api.agnes-ai.cn`，多个用逗号分隔 |
| `AGNES_INTL_API_KEYS` | 否 | - | Agnes 国际站 Key，固定用于 `apihub.agnes-ai.com`，多个用逗号分隔 |
| `CUSTOM_API_KEYS` | 否 | - | 旧版通用客户端 Key，贡献数量固定为 1 |
| `GATEWAY_KEYS_JSON` | 否 | - | 客户端 Key 结构化初始配置，可分别设置各服务商贡献数量和模型白名单 |
| `ADMIN_API_KEY` | 否 | - | `/admin` 独立管理员登录密钥 |
| `UPSTASH_REDIS_REST_URL` | 否 | - | Upstash Redis REST URL（开启云端数据持久化） |
| `UPSTASH_REDIS_REST_TOKEN` | 否 | - | Upstash Redis REST Token |
| `THINKING_MODE` | 否 | `auto` | 思考模式强控：`auto` / `on` / `off` |
| `MODEL_FALLBACK_MODE` | 否 | `auto` | 自动降级模式：`auto` / `off` / `force_gpt` |
| `DEBUG_CAPTURE_PAYLOADS` | 否 | `false` | 是否在 Debug 日志中保存完整请求和响应正文；生产环境建议保持关闭 |

结构化客户端 Key 示例：

```json
{
  "cpr_all_c1_replace-with-random-secret": {
    "name": "通用用户",
    "providers": {
      "cerebras": 1,
      "groq": 1,
      "agnes": 1
    },
    "enabled": true
  },
  "cpr_multi_c3_replace-with-random-secret": {
    "name": "贡献者 A",
    "providers": {
      "cerebras": 2,
      "groq": 1,
      "agnes": 0
    },
    "allowed_models": [
      "gemma-4-31b",
      "gpt-oss-120b",
      "openai/gpt-oss-120b"
    ],
    "enabled": true
  }
}
```

`providers` 中的数量同时表示服务商访问权限和贡献数量。`0` 表示禁止访问该服务商。服务端实际额度始终读取配置内容，不信任 Key 名称中的数字。Admin 页面生成的动态 Key 需要 Upstash 持久化；未配置 Upstash 时 Admin 为只读模式。

Vercel Serverless 的内存和临时文件不会跨实例长期保留。若需要 `/logs`、`/debug`、`/thinkingdisplay`、`/fallbackmode` 在刷新、冷启动和重新部署后保持状态，必须同时配置 `UPSTASH_REDIS_REST_URL` 与 `UPSTASH_REDIS_REST_TOKEN`。

例如贡献者 A 提供了 2 个 Cerebras Key 和 1 个 Groq Key，则配置为 `cerebras: 2`、`groq: 1`、`agnes: 0`。该客户端 Key 可以调用 Cerebras 和 Groq，但 `/v1/models` 不会返回 Agnes 模型，直接请求 Agnes 也会返回 `403 model_not_allowed`。

客户端独立额度基准：Cerebras 为 `5 RPM / 30,000 TPM`，Groq 沿用项目基准 `30 RPM / 40,000 TPM`，Agnes 使用官方按模型/图片分辨率公布的 RPM 且不虚构 TPM。最终额度按对应服务商贡献数量乘以 2。

Agnes 免费/default 官方实际 RPM：文本 20；图片 1K/2K/3K/4K 分别为 20/10/1/1；视频为 1。中国站和国际站分别作为独立限制池，同一站点配置多个 API Key 不会叠加官方 RPM。

视频任务会绑定创建它的客户端 Key 和上游站点，查询结果时必须使用同一个客户端 Key。因此 `/v1/videos` 和视频结果查询接口不支持匿名开放模式。

---

## 📦 全平台部署指南

### 1. 本地运行 (Local Development)

```bash
# 克隆仓库
git clone https://github.com/frozenmooncake/cerebras-proxy-re.git
cd cerebras-proxy-re

# 安装依赖
pip install -r requirements.txt

# 创建 .env 文件并填写变量
echo "CEREBRAS_API_KEYS=csk-your-key-1,csk-your-key-2" > .env

# 启动服务
uvicorn api.main:app --reload --port 8000
```

启动后访问 http://127.0.0.1:8000/menu 即可使用。

### 2. 部署至 [Vercel](https://vercel.com) (Serverless - 推荐)

1. **导入项目**：在 [Vercel](https://vercel.com) 控制台点击 `New Project`，导入本仓库。
2. **框架预设 (Framework Preset)**：选择 `Other`（纯 Python Serverless 项目无需选择前端预设）。
3. **各项设置**：
   - 根目录：保持默认 `./`。
   - 构建指令 / 输出目录：保持留空。
   - 安装命令：保持留空(或填入 `pip install -r requirements.txt`)。
4. **配置环境变量**：点击展开 `Environment Variables`，将 `CEREBRAS_API_KEYS` 等所有密钥逐个添加。
5. **一键部署**：确认无误后，点击底部的 `Deploy` 按钮，等待自动打包上线即可。

### 3. 部署至 [Render](https://render.com) (配合精准保活)

1. 登录 [Render.com](https://render.com) 选择 `New Web Service`，绑定本仓库。
2. **Environment**: `Python 3`
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. 在 `Environment Variables` 填入密钥后发布。
6. **保活策略**：避免 Render 免费额度超标，推荐使用 [cron-job.org](https://cron-job.org) 设置定时任务：
   - Cron 表达式：`*/10 0-1,8-23 * * *`（每 10 分钟请求一次 `/health`，并在夜间 02:00 - 07:59 自动休眠以节省额度）。

### 4. 部署至 [Koyeb](https://www.koyeb.com)

1. 登录 [Koyeb.com](https://www.koyeb.com) 并关联你的 GitHub 仓库。
2. 创建服务，构建类型选择 `Buildpack`。
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. 在 `Environment Variables` 中设置环境变量后点击 `Deploy` 部署。

---

## 🛡️ 高可用双活主备方案 ([Cloudflare Worker](https://www.cloudflare.com))

为了防止单个 Serverless 平台超额限流或冷启动超时，可通过 [Cloudflare Worker](https://www.cloudflare.com)（免费版）实现 [Vercel](https://vercel.com)（主）+ [Render](https://render.com)（备）秒级自动故障转移：

```javascript
export default {
  async fetch(request, env, ctx) {
    const PRIMARY_URL = 'https://your-vercel-app.vercel.app';   // 主节点
    const BACKUP_URL  = 'https://your-render-app.onrender.com'; // 备用节点
    const url     = new URL(request.url);
    const method  = request.method;
    const headers = new Headers(request.headers);
    const body    = (method === 'GET' || method === 'HEAD') ? null : await request.arrayBuffer();

    // 1. 优先尝试主节点 (Vercel)
    try {
      const controller  = new AbortController();
      const timeoutId   = setTimeout(() => controller.abort(), 4000); // 4秒超时自动切备用
      const response    = await fetch(PRIMARY_URL + url.pathname + url.search, {
        method, headers, body, signal: controller.signal
      });
      clearTimeout(timeoutId);
      if (response.status < 500) return response;
    } catch (err) {
      // 主节点超时或异常，静默自动降级至备用节点
    }

    // 2. 主节点不可用，无缝切至备用节点 (Render)
    return fetch(BACKUP_URL + url.pathname + url.search, { method, headers, body });
  }
};
```

---

## 💡 客户端接入说明

以 Cherry Studio / NextChat / LobeChat 等客户端为例：

- **API 协议**：OpenAI
- **API Base URL**：`https://你的接入域名/v1`
- **API Key**：若设置了 `CUSTOM_API_KEYS` 填写对应密钥；未设置可随意填写 `sk-123456`
- **模型列表鉴权**：`GET /v1/models` 使用客户端 Key，不需要 Admin 登录；支持标准 `Authorization: Bearer <Key>`，并兼容 `X-API-Key` / `api-key`
- **可用模型**：
  - `gemma-4-31b`
  - `zai-glm-4.7`
  - `gpt-oss-120b`
  - Groq 模型列表见 `/v1/models`
  - `agnes/agnes-2.5-flash`
  - `agnes/agnes-image-2.1-flash`
  - `agnes/agnes-video-v2.0`

Agnes 付费模型 `agnes-2.5-pro-alpha` 未纳入托管列表。

### 图片输入说明

图像理解支持：

```text
gemma-4-31b
agnes/agnes-2.5-flash
```

云端网关无法读取客户端电脑上的本地路径，例如 `C:\Users\name\image.jpg`。请使用公网可访问的 HTTPS 图片 URL，或客户端上传后生成的 `data:image/...;base64,...` URI。

不支持图片的模型会返回 `image_input_not_supported`；本地路径会返回 `local_image_unavailable`。这两个错误都会以标准 OpenAI JSON 返回，避免第三方客户端只显示“无回复”。

Debug 页面的“一键复制 AI 调试包”会同时复制 `Request Body` 和 `Response Body`。

Agnes 文本模型的流式、非流式和上游错误响应也会写入 `/debug`；流式记录包含生成正文和最多 10 条 SSE Chunk 样例。

---

## 测试

项目测试不依赖额外测试框架：

```bash
python -m unittest discover -s tests -v
```

静态编译检查：

```bash
python -m py_compile api/main.py api/model_catalog.py api/provider_adapters.py api/distributed_limits.py api/access_control.py api/agnes_provider.py api/groq_provider.py
```

### 管理页面安全

以下页面需要先在 `/admin` 使用 `ADMIN_API_KEY` 登录：

```text
/status
/logs
/debug
/config
/thinkingdisplay
/fallbackmode
```

Thinking 和 Fallback 修改已改为 POST 操作，并校验绑定当前 Admin 会话的 CSRF token。`/debug` 默认只保存元数据；仅在受控诊断期间设置 `DEBUG_CAPTURE_PAYLOADS=true`。

---

## 📄 开源许可证

本项目基于 MIT License 许可证开源。
