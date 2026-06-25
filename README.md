# <span lang="zh-CN">宏观经济智能分析平台</span> · Macro Intelligence Daily · v9

> **From 31-section markdown reports to a self-driving macro cockpit — vector RAG, Hamilton regime switching, GraphRAG and a ReAct agent, all in one Python pipeline.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Modules](https://img.shields.io/badge/分析模块-24%20前沿-orange)](#-项目结构)
[![FastAPI](https://img.shields.io/badge/FastAPI-v9-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed?logo=docker)](Dockerfile)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088ff?logo=github-actions)](.github/workflows/ci.yml)
[![LLM](https://img.shields.io/badge/LLM-OpenAI%20compatible-7c3aed)](https://platform.openai.com/docs/api-reference/embeddings)
[![Data: 人民网 + 中国经济网](https://img.shields.io/badge/Data-人民网%20%7C%20中国经济网-c71f1f)](#-项目结构)

每日自动抓取 *人民网财经* 与 *中国经济网* 的宏观新闻 → 跑通 24 个分析模块 → 输出 **31 节结构化报告 + 现代化驾驶舱 + FastAPI REST + 多智能体 + ReAct Agent**,并把每一份产物落盘为可追溯的 Markdown / HTML / SQLite 时序库。

---

## 📑 目录

- [⚡ v9 相对前版更新了什么](#-v9-相对前版更新了什么)
- [✨ 主要能力](#-主要能力)
- [🏗️ 架构总览](#️-架构总览)
- [🚀 快速开始](#-快速开始)
- [🔑 配置你的 API Key](#-配置你的-api-key)
- [🧪 烟测与一键运行](#-烟测与一键运行)
- [🐳 Docker 部署](#-docker-部署)
- [🌐 REST API 速览](#-rest-api-速览)
- [🖥️ 仪表盘与样例报告](#-仪表盘与样例报告)
- [🧩 项目结构](#-项目结构)
- [🛠️ 技术栈](#️-技术栈)
- [❓ 常见问题](#-常见问题)
- [🤝 贡献与许可证](#-贡献与许可证)

---

## ⚡ v9 相对前版更新了什么

> v9 在不臃肿的前提下,把"信号 → 决策 → 解释 → 行动"这条链路彻底打通:向量化语义检索、隐马尔可夫区制、知识图谱、ReAct Agent 第一次同时进入主流程。

| 维度 | v1–v4 (基础) | v5 (NLP + 信号) | v6 (金融工程) | v8 (多因子 + PDF) | **v9 (前沿 AI 量化)** |
|---|---|---|---|---|---|
| 抓取源 | 人民网 | + 中国经济网 | 同 v5 | + 新华 / 央行 可选 | 同 v8 |
| 分析模块 | 6 | 12 | 16 | 19 | **24** |
| AI 维度 | 7 | 10 | 10 | 10 | **10 + LLM-as-Judge** |
| 报告章节 | 14 | 22 | 27 | 31 | **31 + 区制 / Agent / GraphRAG** |
| 仪表盘 | 8 图 | 18 面板 | 18 面板 | 18 面板 | **22 面板 + 多 Tab + SSE** |
| 决策引擎 | — | BUY/HOLD/REDUCE/SELL | + SHAP | + 多因子 | **+ ReAct Agent** |
| 风险 | — | Sharpe/VaR/CVaR | + 8 类 | + 区制 | **+ Hamilton 1989 隐马尔可夫 + 95% 后验** |
| 组合 | — | 行业权重 + 4 调仓 | + 多因子 | + A/B/C/D/E 评级 | **+ 因子归因** |
| 情景 | — | 蒙特卡洛 + 7 情景 | + STL+AR | — | **+ 区制条件蒙特卡洛** |
| 智能问答 | — | RAG + LLM | + embed_rag | — | **+ 向量 RAG (Transformer)** |
| **向量嵌入** | — | TF-IDF | TF-IDF | TF-IDF | **🆕 BERT / Hash 向量 + 余弦检索** |
| **区制切换** | — | — | — | — | **🆕 Hamilton MS-AR (冷静 / 狂热)** |
| **GraphRAG** | — | — | — | — | **🆕 Louvain 社区 + 层级摘要** |
| **ReAct Agent** | — | — | — | — | **🆕 工具调用 Agent (无 LangChain 依赖)** |
| **LLM 评测** | — | self_eval | self_eval | self_eval | **🆕 多 Judge × 多维打分** |
| **REST 端点** | — | — | 22 | 22 | **30+ + SSE 流** |
| **驾驶舱** | — | Plotly 静态 | Plotly 静态 | Plotly 静态 | **🆕 多 Tab + SSE + D3 知识图谱** |
| **Docker / CI** | — | — | ✅ | ✅ | **✅ 镜像 ~280 MB** |

### v9 新增的 5 个前沿模块

- **🧠 `analysis/embeddings.py`** — 中文向量嵌入引擎 (兼容 OpenAI Embedding API),numpy 余弦检索,取代 TF-IDF 的"词汇鸿沟"问题,支持语义级 RAG。带 hash fallback,断网 / 无 HF 权重也能跑。
- **📊 `analysis/regime.py`** — Hamilton (1989) 马尔可夫区制切换 (2 状态: CALM / HOT),纯 numpy 实现,带 EM 估计、平滑后验概率,给市场状态打"冷静 / 狂热"标签。
- **🔍 `analysis/graph_rag.py`** — GraphRAG-lite: 从文章 / 实体 / 关系建图,Louvain 社区发现 + LLM 层级摘要,回答跨文章全局问题 (例如"过去一个月政策主线是什么?")。
- **⚖️ `analysis/llm_judge.py`** — LLM-as-Judge 多维质量评估 (一致性 / 有据性 / 完整性 / 可行动性 / 新颖性),多 Judge 投票,出报告前先自评。
- **🤖 `agent/react_agent.py`** — ReAct 风格 LLM Agent (无 LangChain 依赖),工具集: 信号引擎 / 多因子 / 风险 / 问答 / RAG / 区制 / 情景。给出"思考 → 行动 → 观察"三段式推理。

### v9 工程化提升

- **🌐 现代化驾驶舱 v9** — 多 Tab 导航 (总览 / 信号 / 因子 / 知识图谱 / 问答 / 区制 / 比较),SSE 实时心跳,深色主题 + 渐变卡片。
- **📡 30+ REST 端点** — `/v9/regime` · `/v9/embeddings/search` · `/v9/graphrag/ask` · `/v9/judge` · `/v9/agent/run` · `/v9/stream/*`。
- **🧬 向量化语义搜索** — TF-IDF → 语义向量,跨期主题匹配更准。
- **🔬 多维质量门控** — 4 个 Judge (政策 / 产业 / 市场 / 战略) × 5 维度,出报告前先自评。

---

## ✨ 主要能力

```text
[人民网 + 中国经济网] → [反爬 + 质量过滤 + 语义去重] → [NLP (jieba + TF-IDF + 语义向量)]
                                                                       ↓
[31 节 Markdown 报告]  ← [LLM 反思循环] ← [10 维 LLM 分析 (并发)]
        ↓                                       ↓
[智能驾驶舱仪表盘]                    [24 模块 → SQLite 时序库]
        ↓
[FastAPI REST]  [多智能体顾问团]  [ReAct Agent]  [SHAP 解释]
        ↓
[Prometheus 指标]  [Docker]  [GitHub Actions CI]
```

- 📰 **多源爬虫** — 人民网 + 中国经济网,可选新华网 / 央行;智能反爬 + 质量过滤 + 语义去重。
- 🧠 **10 维 AI 分析** — 主题词 / 政策 / 产业 / 洞察 / 判断 / 立场 / 事件 / 聚类 / 自评 + LLM 反思循环。
- 📊 **24 个前沿模块** — 异常 / 预测 / 产业-A 股 / 宏观 / 主题 / 事件 / RAG / 波动 / 因果 / 信号 / 回测 / 简报 / 问答 / 风险 / 组合 / 情景 / **多因子 / 政策 PDF / 嵌入 / 区制 / GraphRAG / 多智能体 / SHAP / Agent**。
- 🖥️ **现代化驾驶舱** — 22 面板 + Hero 决策卡 + KPI 卡 + 多 Tab + SSE。
- 📜 **31 节结构化报告** — 从核心速览到原始文章清单,每节都可独立追溯。
- 🚀 **FastAPI REST API** — 30+ 端点,Swagger UI 自带。
- 🤖 **多智能体顾问团** — 4 角色 + 仲裁,模板 / LLM 双模式。
- 🔍 **SHAP 风格决策解释** — 特征贡献度,加性分解,闭式可解释。
- 🐳 **Docker / docker-compose / GitHub Actions CI** — 镜像 ~280 MB,启动 < 5 s。

---

## 🏗️ 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│  Data Sources                                                 │
│  · 人民网财经   · 中国经济网   · 新华网(opt)   · 央行(opt)  │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Pipeline  (src/scraper, src/nlp)                            │
│  fetch → parse → quality filter → semantic dedup → jieba    │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  AI Layer  (src/ai) — 10 dimensions + reflection + cache    │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Analysis Layer  (src/analysis) — 24 modules                 │
│  anomaly · forecast · factor · regime · risk · scenario ·    │
│  portfolio · embeddings · graph_rag · llm_judge · ...        │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
  ┌──────────────┐                  ┌──────────────────┐
  │  Reports     │   ◀── LLM Judge ──▶│   Cockpit        │
  │  (md/html)   │                    │   (dashboard/)  │
  └──────────────┘                    └──────────────────┘
        │                                     │
        └──────────────────┬──────────────────┘
                           ▼
              ┌────────────────────────┐
              │  FastAPI + REST + SSE  │
              │  + ReAct Agent + SHAP  │
              └────────────────────────┘
```

---

## 🚀 快速开始

### 0. 环境要求

| 组件 | 版本 |
|---|---|
| Python | 3.11+ (Windows / macOS / Linux) |
| 内存 | ≥ 2 GB 可用 |
| 磁盘 | ≥ 1 GB (含缓存) |
| 网络 | 抓取与 LLM 调用需要出网 |

### 1. 克隆与安装

```bash
git clone https://github.com/<your-account>/people-daily-economy-daily.git
cd people-daily-economy-daily
pip install -r requirements.txt
```

> Windows 上推荐用 `python -X utf8` 启动,避免中文日志乱码。

### 2. 配置 API Key

把 `.env.example` 复制成 `.env`,填入你选用的 LLM 提供方 Key (详见下一节)。

### 3. 一键烟测 (≈ 7 s,无需网络)

```bash
python -X utf8 smoke_test.py --fast
```

### 4. 端到端跑一天

```bash
# 抓取 + 分析 + 报告 + 驾驶舱 (跳过已抓取数据,直接复用缓存)
python -X utf8 main.py --target-date 2026-06-12 --skip-scrape

# 全流程 (抓取 + 分析 + 报告 + 驾驶舱),首次或新一天使用
python -X utf8 main.py --target-date 2026-06-12
```

产物:
- 报告: `reports/2026-06-12.md` (≈ 47 KB) + `reports/2026-06-12.html`
- 驾驶舱: `dashboard/2026-06-12.html`
- 词云 / 知识图谱: `images/wordcloud_2026-06-12.png` · `images/knowledge_graph_2026-06-12.png`
- 时序数据: `data/historical/economy_timeseries.sqlite3`

---

## 🔑 配置你的 API Key

> ⚠️ **本仓库已清空 API Key。** 真实 key 只应放在你自己机器的 `.env` 里,`.env` 已被 `.gitignore` 排除,不会被提交到 GitHub。

把 `.env.example` 复制成 `.env`,按需填写(以下提供方二选一即可):

```bash
# ===== 推荐: OpenAI 兼容协议 (国内直连可选 minimax / DeepSeek / 通义千问) =====
AI_PROVIDER=custom
AI_BASE_URL=https://api.minimaxi.com/v1
AI_MODEL=MiniMax-M3
AI_API_KEY=<你的 key>
```

| 提供方 | 适用 | 推荐模型 | Base URL |
|---|---|---|---|
| **minimax** (推荐) | 国内直连,中文最强 | `MiniMax-M3` (thinking) | `https://api.minimaxi.com/v1` |
| DeepSeek | 性价比高 | `deepseek-chat` | `https://api.deepseek.com/v1` |
| 通义千问 (DashScope) | 阿里云生态 | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| OpenAI | 需代理 | `gpt-4o-mini` | `https://api.openai.com/v1` |

> 💡 **没有 key 也能跑 80% 的功能。** 24 个分析模块里,异常 / 预测 / 波动 / 风险 / 组合 / 情景 / SHAP / 多智能体 都内置了"模板降级"模式,无 LLM 也能出数。LLM 模式只在 `use_llm=True` 时启用。

---

## 🧪 烟测与一键运行

```bash
# 烟测: 7 秒,无网络,无 key
python -X utf8 smoke_test.py --fast

# 端到端 (跳过抓取,复用缓存) — 4-5 分钟
python -X utf8 main.py --target-date 2026-06-12 --skip-scrape

# 端到端 (含抓取) — 8-12 分钟
python -X utf8 main.py --target-date 2026-06-12

# 启动 REST 服务
python -X utf8 -m src.api.server --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs   (Swagger UI)
```

---

## 🐳 Docker 部署

```bash
docker compose up -d
# → http://localhost:8000/docs
```

镜像多阶段构建,体积约 280 MB,启动 < 5 秒。健康检查走 `/health`。

---

## 🌐 REST API 速览

| 模块 | 端点 | 能力 |
|---|---|---|
| Health | `GET /health` | 24 模块健康度 |
| AI | `POST /v6/qa` | RAG 智能问答 |
| AI | `POST /v6/council` | 4 智能体顾问团 + 仲裁 |
| AI | `GET /v6/shap` | SHAP 决策解释 |
| Risk | `GET /v6/risk` | 8 类风险指标 |
| Portfolio | `POST /v6/portfolio` | 行业组合回测 |
| Scenario | `POST /v6/scenario/run` | 蒙特卡洛情景 |
| 🆕 Regime | `GET /v9/regime` | Hamilton 区制后验 |
| 🆕 Embeddings | `POST /v9/embeddings/search` | 语义向量检索 |
| 🆕 GraphRAG | `POST /v9/graphrag/ask` | 跨文章全局问答 |
| 🆕 Judge | `POST /v9/judge` | LLM-as-Judge 自评 |
| 🆕 Agent | `POST /v9/agent/run` | ReAct Agent |
| 🆕 Stream | `GET /v9/stream/*` | SSE 实时流 |

完整列表:启动服务后访问 `http://localhost:8000/docs`。

---

## 🖥️ 仪表盘与样例报告

仓库根目录下的 `dashboard/2026-06-12.html` (≈ 51 KB) 是 22 面板的现代化驾驶舱样例,
`reports/2026-06-12.md` (≈ 47 KB) 是 31 节结构化报告样例 — 直接在浏览器打开即可预览。

| 词云 | 知识图谱 |
|:---:|:---:|
| `images/wordcloud_2026-06-12.png` | `images/knowledge_graph_2026-06-12.png` |

---

## 🧩 项目结构

```text
people-daily-economy-daily/
├── main.py                       # 端到端入口
├── smoke_test.py                 # 24 模块烟测 (≈ 7 s,无 key)
├── requirements.txt
├── Dockerfile · docker-compose.yml
├── .env.example                  # ← 复制为 .env 并填 key
├── .github/workflows/ci.yml      # GitHub Actions CI
├── src/
│   ├── scraper/                  # 多源抓取 + 反爬 + 语义去重
│   ├── nlp/                      # jieba + TF-IDF + 词云
│   ├── ai/                       # 10 维 LLM 分析 + 反思 + 缓存
│   ├── analysis/                 # 24 个分析模块 (v1–v9 累加)
│   ├── agent/                    # 🆕 ReAct Agent
│   ├── kg/                       # 知识图谱
│   ├── dashboard/                # 驾驶舱模板
│   ├── report/                   # 报告模板 (Markdown / HTML)
│   ├── api/                      # FastAPI + SSE + 多智能体 + SHAP
│   ├── storage/                  # SQLite 时序库
│   └── utils/
├── data/{raw,processed,historical,cache}/
├── reports/                      # 历史报告 (按日期)
├── dashboard/                    # 历史驾驶舱 (按日期)
├── images/                       # 词云 / 知识图谱
├── notebooks/                    # Jupyter 探索
└── scripts/                      # 维护脚本
```

---

## 🛠️ 技术栈

| 层 | 选型 |
|---|---|
| 抓取 | `requests` · `BeautifulSoup4` · `lxml` · 自研反爬 / 质量过滤 / 语义去重 |
| NLP | `jieba` · `scikit-learn` (TF-IDF / LDA) · `wordcloud` · `matplotlib` |
| 量化 | `numpy` · `scipy` · `networkx` · 自研 Hamilton MS-AR / SHAP 风格 / 蒙特卡洛 |
| LLM | OpenAI 兼容协议 (支持 minimax / DeepSeek / 通义千问 / OpenAI) |
| 嵌入 | `torch` · `transformers` (中文向量,带 hash fallback) |
| API | `FastAPI` · `uvicorn` · `pydantic v2` · `httpx` · SSE |
| 工程 | `python-dotenv` · `Jinja2` · `prometheus-client` · `Docker` · `GitHub Actions` |

---

## ❓ 常见问题

**Q1: 一定要 API key 吗?**
A: 不必须。LLM 用于"10 维语义分析 + RAG 问答 + 多智能体 + Agent + LLM Judge",其余 80% 模块 (异常 / 预测 / 风险 / 组合 / 情景 / 区制 / SHAP / 词云 / 知识图谱) 都内置模板降级。

**Q2: 我能换数据源吗?**
A: 可以。`src/scraper/multi_source.py` 中注册新的 `BaseSource` 子类即可,默认 人民网 + 中国经济网,可启用 `ENABLE_XINHUA=1` / `ENABLE_GOV_PBOC=1`。

**Q3: v9 的"向量嵌入"必须要 transformers 吗?**
A: 不必须。代码自动检测环境,有 HF 权重用 BERT,没有则 fallback 到哈希向量 (`prefer="hash"`),保证不卡下载。

**Q4: 烟测 / 端到端跑多久?**
A: 烟测 ~7 s,端到端 (skip-scrape) ~4-5 分钟,端到端 (含抓取) 8-12 分钟。LLM 步骤并发 4-8 路,失败会重试 + 反思 1 轮。

**Q5: Windows 上中文乱码怎么办?**
A: 所有命令加 `python -X utf8` 前缀;`.env` 里 `PYTHONIOENCODING=utf-8` 也已注释提示。

**Q6: Docker 镜像多大?**
A: 多阶段构建后 ~280 MB,启动 < 5 s,健康检查走 `/health`。

**Q7: 我能扩展新情景 / 新因子吗?**
A: 能。`src/analysis/scenario.py` 顶部 `SCENARIOS` dict 加 shock / vol_mult / horizon,`src/analysis/factor_model.py` 顶部 `FACTOR_LIBRARY` 加新因子,均无需重启服务。

**Q8: OpenAPI 文档在哪?**
A: 启动 `src.api.server` 后 `http://localhost:8000/docs` (Swagger UI) / `/redoc` (ReDoc)。

---

## 🤝 贡献与许可证

欢迎 PR / Issue。在提 PR 前请:

1. 运行 `python -X utf8 smoke_test.py --fast` 确保烟测全绿。
2. 若新增 LLM 任务,补 `src/ai/prompts.py` 与 `src/ai/schema.py`。
3. 若新增分析模块,补 `src/analysis/` 下独立文件 + 在 `main.py` 串接。

本项目基于 **MIT License** 开源,详见 [LICENSE](LICENSE)。

### 🙏 致谢

数据源: [人民网财经](http://finance.people.com.cn) · [中国经济网](http://www.ce.cn)
LLM 提供方: [minimax](https://api.minimaxi.com) · [DeepSeek](https://deepseek.com) · [通义千问](https://tongyi.aliyun.com) · [OpenAI](https://openai.com)

### ⚠️ 免责声明

本项目仅作技术研究与学习用途,所产出的分析与建议**不构成任何投资建议**。请使用者自行判断与承担风险。
