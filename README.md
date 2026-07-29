# Cerebras OpenAI API Gateway 2.0.4

> 极速、高可用、多 Key 轮询与智能降级的 Cerebras 到 OpenAI 格式 API 转接网关。

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
- **思考过程深度控制 (Thinking Control)**：支持 `AUTO` / `ON` / `OFF` 三档控制，可一键全局强行抹除思考推理过程，极致节省 Token 消耗与响应耗时。
- **可视化监控与深度调试**：提供极简黑夜风格看板，实时查看所有 Key 的 RPM/RPD/TPM/TPD 水位线、最近 100 条请求历史，以及支持同时打包 Request 与 Response 的一键调试包复制。
- **OpenCode / Cursor 兼容**：补齐 OpenAI 标准模型列表与流式字段，便于编辑器直接识别并接入。
- **云端/本地双模持久化**：原生支持 Upstash Redis 异步持久化存储统计与 Key 池状态；无 Redis 时自动平滑回退至本地 JSON 文件存储。

---

## 🎛️ 可视化控制面板

部署成功后，访问网关根路径或相应路由即可进入管理界面：

| 路由页面 | 功能说明 |
| :--- | :--- |
| `/menu` | 网关主菜单与全局配置概览 |
| `/status` | 实时全局负载水位、上游限额看板与物理 Key 冷却状态 |
| `/thinkingdisplay` | 强行控制/抹除 AI 思考推理过程 (`AUTO` / `ON` / `OFF`) |
| `/fallbackmode` | GLM -> GPT 自动降级策略切换 (`AUTO` / `OFF` / `FORCE_GPT`) |
| `/log` | 最近 100 条请求日志回溯 |
| `/debug` | 最近 50 条全量 Request/Response 抓包与一键打包复制 AI 调试信息 |

---

## ⚙️ 环境变量配置

在部署平台（或本地 `.env` 文件）中配置以下环境变量：

| 环境变量名 | 必填 | 默认值 | 说明 |
| :--- | :---: | :---: | :--- |
| `CEREBRAS_API_KEYS` | **是** | - | 物理 Cerebras API Key，多个用英文逗号分隔，如 `csk-key1,csk-key2` |
| `GROQ_API_KEYS` | 否 | - | Groq API Key，多个用英文逗号分隔，如 `gsk-key1,gsk-key2` |
| `CUSTOM_API_KEYS` | 否 | - | 自定义客户端鉴权 Key（设置后客户端请求必须携带 `Bearer <Key>`） |
| `UPSTASH_REDIS_REST_URL` | 否 | - | Upstash Redis REST URL（开启云端数据持久化） |
| `UPSTASH_REDIS_REST_TOKEN` | 否 | - | Upstash Redis REST Token |
| `THINKING_MODE` | 否 | `auto` | 思考模式强控：`auto` / `on` / `off` |
| `MODEL_FALLBACK_MODE` | 否 | `auto` | 自动降级模式：`auto` / `off` / `force_gpt` |

---

## 📦 全平台部署指南

### 1. 本地运行 (Local Development)

```bash
# 克隆仓库
git clone https://github.com/xyrct301/cerebras-proxy-re.git
cd cerebras-proxy-re

# 安装依赖
pip install -r requirements.txt

# 创建 .env 文件并填写变量
echo "CEREBRAS_API_KEYS=csk-your-key-1,csk-your-key-2" > .env

# 启动服务
uvicorn api.main:app --reload --port 8000
```

启动后访问 http://127.0.0.1:8000/menu 即可使用。

### 2. 部署至 Vercel (Serverless - 推荐)

1. **导入项目**：在 Vercel 控制台点击 `New Project`，导入本仓库。
2. **框架预设 (Framework Preset)**：选择 `Other`（纯 Python Serverless 项目无需选择前端预设）。
3. **各项设置**：
   - 根目录：保持默认 `./`。
   - 构建指令 / 输出目录：保持留空。
   - 安装命令：保持留空(或填入 `pip install -r requirements.txt`)。
4. **配置环境变量**：点击展开 `Environment Variables`，将 `CEREBRAS_API_KEYS` 等所有密钥逐个添加。
5. **一键部署**：确认无误后，点击底部的 `Deploy` 按钮，等待自动打包上线即可。

### 3. 部署至 Render (配合精准保活)

1. 登录 Render.com 选择 `New Web Service`，绑定本仓库。
2. **Environment**: `Python 3`
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. 在 `Environment Variables` 填入密钥后发布。
6. **保活策略**：避免 Render 免费额度超标，推荐使用 cron-job.org 设置定时任务：
   - Cron 表达式：`*/10 0-1,8-23 * * *`（每 10 分钟请求一次 `/health`，并在夜间 02:00 - 07:59 自动休眠以节省额度）。

### 4. 部署至 Koyeb

1. 登录 Koyeb.com 并关联你的 GitHub 仓库。
2. 创建服务，构建类型选择 `Buildpack`。
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. 在 `Environment Variables` 中设置环境变量后点击 `Deploy` 部署。

---

## 🛡️ 高可用双活主备方案 (Cloudflare Worker)

为了防止单个 Serverless 平台超额限流或冷启动超时，可通过 Cloudflare Worker（免费版）实现 Vercel（主）+ Render（备）秒级自动故障转移：

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
- **可用模型**：
  - `gemma-4-31b`
  - `zai-glm-4.7`
  - `gpt-oss-120b`
  - Groq 模型列表见 `/v1/models`

---

## 📄 开源许可证

本项目基于 MIT License 许可证开源。
