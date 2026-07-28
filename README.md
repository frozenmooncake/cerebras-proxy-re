<div align="center">

# 🧠 Cerebras OpenAI API Gateway 2.0

> 🚀 极速、高可用、多 Key 轮询与智能降级的 Cerebras 到 OpenAI 格式 API 转接网关。

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![HTTP2 Enabled](https://img.shields.io/badge/HTTP%2F2-Enabled-brightgreen)

[功能特性](#-功能特性) •
[部署指南](#-全平台部署指南) •
[控制面板](#-可视化控制面板) •
[环境变量](#-环境变量配置) •
[客户端接入](#-客户端接入说明)

</div>

---

## 🌟 功能特性

* ⚡ **极速流式响应 (Stream Optimization)**：采用 **FastAPI + HTTP/2 Multiplexing + 长连接复用**，大幅降低首字延迟（TTFT），支持毫秒级打字机推流。
* 🔑 **多物理 Key 智能轮询 (Key Pool)**：支持配置多个 Cerebras API-Key，自动负载均衡、计算限额并在遇到 `429 Too Many Requests` 时自动触发 60s 智能冷却隔离。
* 🔀 **智能模型自动降级 (Fallback System)**：当上游 GLM 模型 (`zai-glm-4.7`) 发生限流或异常时，可无缝无感自动降级至 GPT 模型 (`gpt-oss-120b`) 接管响应。
* 🎯 **思考过程深度控制 (Thinking Control)**：支持 `AUTO` / `ON` / `OFF` 三档控制，可一键全局强行抹除思考推理过程，极致节省 Token 消耗与响应耗时。
* 📊 **可视化监控与深度调试**：提供极简黑夜风格看板，实时查看所有 Key 的 RPM/RPD/TPM/TPD 水位线、最近 100 条请求历史及全量 Debug 抓包。
* 💾 **云端/本地双模持久化**：原生支持 Upstash Redis 异步持久化存储统计与 Key 池状态；无 Redis 时自动平滑回退至本地 JSON 文件存储。

---

## 🎛️ 可视化控制面板

部署成功后，访问网关根路径或路由即可进入管理界面：

| 路由页面 | 功能说明 |
| :--- | :--- |
| `/menu` | 🏠 网关主菜单与全局配置概览 |
| `/status` | 📊 实时全局负载水位、上游限额看板与物理 Key 冷却状态 |
| `/thinkingdisplay` | 🎯 强行控制/抹除 AI 思考推理过程 (`AUTO` / `ON` / `OFF`) |
| `/fallbackmode` | 🔀 GLM -> GPT 自动降级策略切换 (`AUTO` / `OFF` / `FORCE_GPT`) |
| `/log` | 📜 最近 100 条请求日志回溯 |
| `/debug` | 🔍 最近 50 条全量 Request/Response 抓包与调试包复制 |

---

## 🛠️ 环境变量配置

在部署平台（或本地 `.env` 文件）中配置以下环境变量：

| 环境变量名 | 必填 | 默认值 | 说明 |
| :--- | :---: | :---: | :--- |
| `CEREBRAS_API_KEYS` | **是** | - | 物理 Cerebras API Key，多个用英文逗号分隔，如 `csk-key1,csk-key2` |
| `CUSTOM_API_KEYS` | 否 | - | 自定义客户端鉴权 Key（设置后客户端请求必须携带 `Bearer <Key>`） |
| `UPSTASH_REDIS_REST_URL` | 否 | - | Upstash Redis REST URL（开启云端数据持久化） |
| `UPSTASH_REDIS_REST_TOKEN` | 否 | - | Upstash Redis REST Token |
| `THINKING_MODE` | 否 | `auto` | 思考模式强控：`auto` / `on` / `off` |
| `MODEL_FALLBACK_MODE` | 否 | `auto` | 自动降级模式：`auto` / `off` / `force_gpt` |

---

## 🚀 全平台部署指南

### 1. 本地运行 (Local Development)

```bash
# 1. 克隆仓库
git clone https://github.com/xyrct301/cerebras-proxy-re.git
cd cerebras-proxy-re

# 2. 安装依赖
pip install -r requirements.txt

# 3. 创建 .env 文件并填写变量
echo "CEREBRAS_API_KEYS=csk-your-key-1,csk-your-key-2" > .env

# 4. 启动服务
uvicorn api.index:app --reload --port 8000
```

访问 `http://127.0.0.1:8000/menu` 即可使用。

---

### 2. 部署至 Vercel (Serverless)

仓库内置了 `vercel.json` 配置文件：

```json
{
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/api/index"
    }
  ]
}

直接点击导入 Vercel 项目，在 Settings -> Environment Variables 注入配置即可一键部署。

---

### 3. 部署至 Render (配合保活)

1. 登录 [Render.com](https://render.com) 选择 **New Web Service**，绑定本仓库。
2. **Environment**: `Python 3`
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn api.index:app --host 0.0.0.0 --port $PORT`
5. 在 **Environment Variables** 填入密钥后发布。
6. *提示：为防止 Render 15 分钟无流量休眠，可用 [UptimeRobot](https://uptimerobot.com) 每 10 分钟 Ping 一次 `https://your-app.onrender.com/health`。*

---

### 4. 部署至 Koyeb

1. 登录 [Koyeb.com](https://www.koyeb.com) 并关联你的 GitHub 仓库。
2. 创建服务，构建类型选择 **Buildpack** 或 **Docker**。
3. 暴露端口设置：`8000`。
4. 在 **Environment Variables** 中设置 `CEREBRAS_API_KEYS` 等配置。
5. 点击 **Deploy** 即可完成公网上线。

---
```

## 💻 客户端接入说明

以 **Cherry Studio / NextChat / LobeChat** 等客户端为例：

* **API 协议**：`OpenAI`
* **API Base URL**：`https://你的域名/v1`
* **API Key**：如果你设置了 `CUSTOM_API_KEYS` 则填写对应密钥；若未设置可随意填 `sk-123456`
* **可用模型**：
    * `zai-glm-4.7`（支持自动降级至 GPT）
    * `gpt-oss-120b`

---

## 📄 开源许可证

本项目基于 [MIT License](LICENSE) 许可证开源。
