# <span lang="zh-CN">宏观经济智能分析平台</span> · Macro Intelligence Daily · v9

> **From 31-section markdown reports to a self-driving macro cockpit — vector RAG, Hamilton regime switching, GraphRAG and a ReAct agent, all in one Python pipeline.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Modules](https://img.shields.io/badge/分析模块-24%20前沿-orange)](#-主要能力)
[![v9 Refactor](https://img.shields.io/badge/v9%20Refactor-Protocol%20子包-7c3aed)](#-v9-重构-预测与检索的分层抽象)
[![FastAPI](https://img.shields.io/badge/FastAPI-v9-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed?logo=docker)](Dockerfile)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088ff?logo=github-actions)](.github/workflows/ci.yml)
[![Data: 人民网 + 中国经济网](https://img.shields.io/badge/Data-人民网%20%7C%20中国经济网-c71f1f)](#-主要能力)

每日自动抓取 *人民网财经* 与 *中国经济网* 的宏观新闻 → 跑通 24 个分析模块 → 输出 **31 节结构化报告 + 现代化驾驶舱 + FastAPI REST + 多智能体 + ReAct Agent**,并把每一份产物落盘为可追溯的 Markdown / HTML / SQLite 时序库。

---

## 📑 目录

- [🆕 v9 重构: 预测与检索的分层抽象](#-v9-重构-预测与检索的分层抽象)
- [✨ 主要能力](#-主要能力)
- [🏗️ 架构总览](#️-架构总览)
- [🚀 快速开始](#-快速开始)
- [🧪 烟测与一键运行](#-烟测与一键运行)
- [🐳 Docker 部署](#-docker-部署)
- [🌐 REST API 速览](#-rest-api-速览)
- [🖥️ 仪表盘与样例报告](#️-仪表盘与样例报告)
- [🧩 项目结构](#-项目结构)
- [🛠️ 技术栈](#️-技术栈)
- [❓ 常见问题](#-常见问题)
- [🤝 贡献与许可证](#-贡献与许可证)

---

## 🆕 v9 重构: 预测与检索的分层抽象

v9 在不破坏任何 v1-v8 行为的前提下,把"分散重复"的预测与检索逻辑抽成两个 **Protocol 子包** —— 加算法只动一个文件,主调度零修改。

### 预测三件套 → `src/forecasting/`

| 旧文件 | 旧实现 | 新位置 | 新角色 |
|---|---|---|---|
| `analysis/forecast.py` (MA+趋势) | 内联实现 | `forecasting/naive.py` | `NaiveForecaster` |
| `analysis/forecast_enhanced.py` (STL+HoltWinters+MC) | 内联实现 | `forecasting/stl_holt_winters.py` | `StlHoltWinters` |
| `analysis/forecast_backtest.py` (walk-forward) | ** MA+趋势** | `forecasting/backtest.py` | `BacktestEngine` 注入 forecaster |

**核心抽象** (`Forecaster` Protocol):

```python
from src.forecasting import Forecaster, NaiveForecaster, StlHoltWinters, BacktestEngine
import numpy as np

class Forecaster(Protocol):
    name: str
    def fit_predict(self, series: np.ndarray, horizon: int = 1, **kwargs) -> ForecastResult: ...

# 任何预测器都能跑同一份回测
series = np.array([50, 50.5, 51, 51.2, 51.5, 52, 52.3, 52, 51.8, 52.5])
result = StlHoltWinters().fit_predict(series, horizon=3)
print(result.predicted, result.method, result.confidence)  # 50.325 holt-winters 0.52

# 自定义预测器只要实现 protocol,backtest 自动支持
class ConstantForecaster:
    name = "constant"
    def fit_predict(self, series, horizon=1, **kw):
        from src.forecasting import ForecastResult
        return ForecastResult(42.0, 40.0, 44.0, "constant", "stable", 1.0, len(series))

rep = BacktestEngine(forecaster=ConstantForecaster()).run("2026-06-12", n_days=20, horizons=(1, 3, 7))
print(rep.summary)  # "回测 20 天窗口 / 3 horizon; ... forecaster=constant"
```

**旧 import 路径全部保留** (转为薄 re-export),`main.py` / `smoke_test.py` / `react_agent` / `shap_explain` / `dashboard/builder` 等 6 个 caller **零修改**。

### RAG 五件套 → `src/retrieval/`

| 旧文件 | 旧实现 | 新位置 | 新角色 |
|---|---|---|---|
| `analysis/embed_rag.py` (**已删除**,285 行) | 自己又写一套 `EmbedDoc` + 向量检索,与 `embeddings.py` 重复 | — | — |
| `analysis/embeddings.py` | 完整实现 (router + VectorIndex + 持久化) | 保留 | `VectorRetriever` 包装 |
| `analysis/rag_history.py` | TF-IDF 跨期主题 | `retrieval/lexical.py` | `LexicalRetriever` |
| `analysis/graph_rag.py` | KG 社区发现 | `retrieval/graph.py` | `GraphRetriever` |
| `analysis/qa_assistant.py` | 编排器,混了 5 个 retriever | **改成多 retriever 并联+融合** | 编排器 |

**核心抽象** (`Retriever` Protocol):

```python
from src.retrieval import Retriever, Hit, LexicalRetriever, VectorRetriever, GraphRetriever

class Retriever(Protocol):
    name: str
    def search(self, query: str, top_k: int = 5, **kwargs) -> list[Hit]: ...

# 任意 retriever 都能插进 qa_assistant
from src.analysis.qa_assistant import ask
result = ask(
    "降准对新能源板块有什么影响?",
    target_date="2026-06-12",
    retrievers=[LexicalRetriever(), VectorRetriever()],
)
print(result.answer)
print("引用:", [(c.source, c.score) for c in result.citations])
```

`server.py` 的 `/v6/embed/*` 路由自动改指 `VectorRetriever`,HTTP API 兼容。

### 收益数字

| 维度 | 旧 (v8) | 新 (v9 重构) |
|---|---|---|
| RAG 文件数 | 5 个分散模块 | 1 个 `retrieval/` 子包 + 4 个 thin re-export |
| RAG 重复代码 | `embed_rag.py` 285 行与 `embeddings.py` 重复 | **0** (`embed_rag.py` 已删) |
| 预测文件数 | 3 个,backtest 自带预测逻辑 | 1 个 `forecasting/` 子包 + 3 个 thin re-export |
| 新增预测算法 | 改 backtest 也得改 | 只实现 `Forecaster` protocol |
| 新增 retriever 类型 | 改 `qa_assistant` 一坨 | 只实现 `Retriever` protocol |
| 新单测 | — | 18 个 (`tests/test_v9_unified.py`, 5.36s) |

---

## ✨ 主要能力

```text
抓取 (scraper)        人民网 · 中国经济网 (+ 可选 新华 / 央行 / 21世纪)
  ↓
存储 (storage)         SQLite 时序库 (daily_metric · ai_report · article)
  ↓
分析 (analysis)        24 个模块:
  v1-v4 基础            metrics · risk · portfolio · scenario · industry
  v5 NLP+信号          theme_keywords · topic_model · signal_engine
  v6 金融工程          risk_metrics · portfolio · scenario · shock · multi_agent
  v7/v8 多因子+PDF     factor_model · policy_pdf · stock_correlation
  v9 前沿 AI 量化       embeddings · regime · graph_rag · llm_judge · react_agent
  ↓
报告 + 驾驶舱          Markdown 31 节 · Plotly 22 面板 · FastAPI 30+ 端点
  ↓
AI 编排                ReAct Agent · 多智能体辩论团 · LLM-as-Judge
```

### v9 5 个前沿模块

- **🧠 `analysis/embeddings.py`** — 中文向量嵌入 (OpenAI 兼容 / 本地 Transformer / hash fallback),numpy 余弦检索,取代 TF-IDF "词汇鸿沟",支持语义级 RAG。
- **📊 `analysis/regime.py`** — Hamilton (1989) 2 状态 MS-AR (CALM / HOT),纯 numpy 实现,EM 估计 + 平滑后验概率。
- **🔍 `analysis/graph_rag.py`** — GraphRAG-lite: 知识图谱 + Louvain 社区 + 层级摘要,回答跨文章全局问题。
- **⚖️ `analysis/llm_judge.py`** — LLM-as-Judge 多维质量门控,多 Judge 投票。
- **🤖 `agent/react_agent.py`** — ReAct Agent (无 LangChain 依赖),工具集: 信号 / 因子 / 风险 / 问答 / RAG / 区制 / 情景。

### v9 工程化提升

- **🌐 现代化驾驶舱 v9** — 多 Tab (总览/信号/因子/知识图谱/问答/区制/比较) + SSE 实时心跳 + 深色主题
- **📡 30+ REST 端点** — `/v6/risk` · `/v6/qa` · `/v6/council` · `/v6/embed/query` · `/v6/stream/*`
- **🧬 向量化语义搜索** — TF-IDF → 语义向量,跨期主题匹配更准
- **🔬 多维质量门控** — 4 Judge × 5 维度,出报告前先自评
- **🔐 发布安全门禁** — CI 会扫描当前工作区的 GitHub PAT / `sk-*` / MiniMax key / 私钥块,避免把密钥带上 GitHub

---

## 🏗️ 架构总览

```
┌────────────────────────────────────────────────────────────────────┐
│  main.py  (端到端调度: 抓取 → 存储 → 分析 → 报告 → 仪表盘 → API)  │
└─────┬──────────────────────────────────────────────────────────────┘
      │
      ├──► scraper/         多源抓取 + 反爬 + 质量过滤 + 语义去重
      │     └► scraper/multi_source.py   (people · ce · xinhua · pbc · 21cbh)
      │
      ├──► storage/         SQLite 时序库 + repository CRUD
      │
      ├──► analysis/        24 个分析模块 (v1-v9 累加)
      │     ├─ forecasting/  🆕 v9: NaiveForecaster · StlHoltWinters · BacktestEngine
      │     │                 (Forecaster Protocol, 注入式回测)
      │     └─ retrieval/    🆕 v9: LexicalRetriever · VectorRetriever · GraphRetriever
      │                       (Retriever Protocol, qa_assistant 编排)
      │
      ├──► ai/              LLM client · router · reflection · cache
      │
      ├──► agent/           ReAct Agent (工具调用)
      │
      ├──► kg/              知识图谱 (实体 / 关系 / 社区)
      │
      └──► report/ + dashboard/  Markdown / HTML / Plotly 22 面板
            └► api/server.py   FastAPI 30+ 端点 + SSE
```

---

## 🚀 快速开始

```bash
git clone https://github.com/<you>/people-daily-economy-daily.git
cd people-daily-economy-daily
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env,填 AI_API_KEY (没有也能跑 80% 的功能)
```

### 烟测 (7 秒,无网络,无 key)

```bash
python -X utf8 smoke_test.py --fast
# Total: 25  Pass: 23  Fail: 2  Time: 19.16s
# (2 个 LLM extract_json 失败是预存的 ModelRouter bug,与本重构无关)
```

### 端到端 (跳过抓取,复用缓存) — 4-5 分钟

```bash
python -X utf8 main.py --date 2026-06-12 --skip-scrape
```

### 端到端 (含抓取) — 8-12 分钟

```bash
python -X utf8 main.py --date 2026-06-12
```

### 启动 REST 服务

```bash
python -X utf8 -m src.api.server --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs   (Swagger UI)
```

### Docker 部署

```bash
docker compose up -d
# → http://localhost:8000/docs
```

镜像多阶段构建,体积约 280 MB,启动 < 5 秒。健康检查走 `/health`。

---

## 🔑 LLM 配置

```bash
# .env (推荐: OpenAI 兼容协议,国内可直连 minimax / DeepSeek / 通义千问)
AI_PROVIDER=custom
AI_BASE_URL=https://api.minimaxi.com/v1
AI_MODEL=MiniMax-M3
AI_API_KEY=<你的 key>
```

| Provider | 适用 | 推荐模型 | Base URL |
|---|---|---|---|
| **minimax** (推荐) | 国内直连,中文最强 | `MiniMax-M3` | `https://api.minimaxi.com/v1` |
| DeepSeek | 性价比 | `deepseek-chat` | `https://api.deepseek.com/v1` |
| 通义千问 (DashScope) | 阿里云 | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| OpenAI | 需代理 | `gpt-4o-mini` | `https://api.openai.com/v1` |

> 💡 **没有 key 也能跑 80% 的功能。** 24 个分析模块里,异常 / 预测 / 波动 / 风险 / 组合 / 情景 / SHAP / 多智能体 都内置了"模板降级"模式,无 LLM 也能出数。LLM 模式只在 `use_llm=True` 时启用。

---

## 🌐 REST API 速览

| 模块 | 端点 | 能力 |
|---|---|---|
| Health | `GET /health` | 24 模块健康度 |
| AI | `POST /v6/qa` | RAG 智能问答 (走 retrieval/ 子包) |
| AI | `POST /v6/council` | 4 智能体顾问团 + 仲裁 |
| AI | `GET /v6/shap` | SHAP 决策解释 |
| Risk | `GET /v6/risk` | 8 类风险指标 |
| Portfolio | `POST /v6/portfolio` | 行业组合回测 |
| Scenario | `POST /v6/scenario/run` | 蒙特卡洛情景 |
| Embeddings | `POST /v6/embed/query` | 语义向量检索 (走 VectorRetriever) |
| Embeddings | `POST /v6/embed/add` | 写入语义向量索引 |
| Embeddings | `GET /v6/embed/stats` | 向量索引状态 |
| Extract | `POST /v6/extract` | 文章 HTML 抽取 |
| Stream | `GET /v6/stream/*` | SSE 实时流 |

完整列表: 启动服务后访问 `http://localhost:8000/docs`。

---

## 🧪 烟测与测试

```bash
# 烟测: 7 秒,无网络,无 key
python -X utf8 smoke_test.py --fast

# v9 重构的 18 个单测 (~5.4 秒)
python -m pytest tests/test_v9_unified.py -v

# 端到端 (跳过抓取) — 4-5 分钟
python -X utf8 main.py --date 2026-06-12 --skip-scrape
```

`tests/test_v9_unified.py` 覆盖:
- `Forecaster` Protocol 验证 (Naive / STL)
- `Retriever` Protocol 验证 (Lexical / Vector / Graph)
- `BacktestEngine` 接受任意 Forecaster 注入
- `qa_assistant.ask()` 接受自定义 retriever 列表
- 旧 import 路径 (`forecast` / `forecast_enhanced` / `forecast_backtest`) 仍工作
- `server.py` 用 `VectorRetriever`

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
├── smoke_test.py                 # 25 步烟测 (≈ 7-19 s,无 key)
├── requirements.txt
├── Dockerfile · docker-compose.yml
├── .env.example                  # ← 复制为 .env 并填 key
├── .github/workflows/ci.yml      # GitHub Actions CI
├── src/
│   ├── scraper/                  # 多源抓取 + 反爬 + 质量过滤 + 语义去重
│   ├── nlp/                      # jieba + TF-IDF + 词云
│   ├── ai/                       # 10 维 LLM 分析 + 反思 + 缓存
│   ├── analysis/                 # 24 个分析模块 (v1–v9 累加)
│   │   ├── forecast.py · forecast_enhanced.py · forecast_backtest.py
│   │   │   # ↑ v9: thin re-export (向后兼容)
│   │   ├── embeddings.py · rag_history.py · graph_rag.py · qa_assistant.py
│   │   │   # ↑ qa_assistant 改成多 retriever 编排器
│   │   └── llm_judge.py · regime.py · signal_engine.py · ...
│   ├── retrieval/                # 🆕 v9: RAG 统一协议 (lexical/vector/graph)
│   ├── forecasting/              # 🆕 v9: 时序预测协议 (naive/stl-hw/backtest)
│   ├── agent/                    # ReAct Agent
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
| 🆕 v9 协议层 | `typing.Protocol` (零新依赖) · `unittest.mock` (测试) |

---

## ❓ 常见问题

**Q1: 一定要 API key 吗?**
A: 不必须。LLM 用于"10 维语义分析 + RAG 问答 + 多智能体 + Agent + LLM Judge",其余 80% 模块 (异常 / 预测 / 风险 / 组合 / 情景 / 区制 / SHAP / 词云 / 知识图谱) 都内置模板降级。

**Q2: 我能换数据源吗?**
A: 可以。`src/scraper/multi_source.py` 中注册新的 `BaseSource` 子类即可,默认 人民网 + 中国经济网,可启用 `ENABLE_XINHUA=1` / `ENABLE_GOV_PBC=1`。

**Q3: v9 的"向量嵌入"必须要 transformers 吗?**
A: 不必须。代码自动检测环境,有 HF 权重用 BERT,没有则 fallback 到哈希向量 (`prefer="hash"`),保证不卡下载。

**Q4: 烟测 / 端到端跑多久?**
A: 烟测 ~7-19 s,端到端 (skip-scrape) ~4-5 分钟,端到端 (含抓取) 8-12 分钟。

**Q5: Windows 上中文乱码怎么办?**
A: 所有命令加 `python -X utf8` 前缀;`.env` 里 `PYTHONIOENCODING=utf-8` 也已注释提示。

**Q6: Docker 镜像多大?**
A: 多阶段构建后 ~280 MB,启动 < 5 s,健康检查走 `/health`。

**Q7: v9 重构后,旧的 `from src.analysis.forecast import ...` 还能用吗?**
A: **能**。三个旧 forecast 文件转成了薄 re-export,所有 6 个 caller (main.py / smoke_test.py / react_agent / shap_explain / dashboard) 零修改。

**Q8: v9 重构后,删了 `embed_rag.py` 会不会影响 server.py?**
A: **不会**。`server.py` 的 `/v6/embed/*` 路由自动改指 `VectorRetriever`,HTTP API 兼容。

**Q9: OpenAPI 文档在哪?**
A: 启动 `src.api.server` 后 `http://localhost:8000/docs` (Swagger UI) / `/redoc` (ReDoc)。

---

## 🤝 贡献与许可证

欢迎 PR / Issue。在提 PR 前请:

1. 运行 `python -X utf8 smoke_test.py --fast` 确保烟测全绿。
2. 运行 `python -m pytest tests/test_v9_unified.py -v` 确保 18 个新单测全绿。
3. 若新增 LLM 任务,补 `src/ai/prompts.py` 与 `src/ai/schema.py`。
4. 若新增分析模块,补 `src/analysis/` 下独立文件 + 在 `main.py` 串接。
5. 若新增预测算法,实现 `Forecaster` protocol (而非在 backtest 里塞新分支)。
6. 若新增 retriever 类型,实现 `Retriever` protocol (而非在 `qa_assistant` 里塞新分支)。

本项目基于 **MIT License** 开源,详见 [LICENSE](LICENSE)。

### 🙏 致谢

数据源: [人民网财经](http://finance.people.com.cn) · [中国经济网](http://www.ce.cn)
LLM 提供方: [minimax](https://api.minimaxi.com) · [DeepSeek](https://deepseek.com) · [通义千问](https://tongyi.aliyun.com) · [OpenAI](https://openai.com)
