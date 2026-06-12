# people-daily-economy-daily
# 人民日报 + 中国经济网 · 经济新闻每日热点报告

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![Requests](https://img.shields.io/badge/Requests-2.31%2B-2C7BB6?logo=python)](https://requests.readthedocs.io/)
[![Jieba](https://img.shields.io/badge/Jieba-0.42%2B-orange)](https://github.com/fxsjy/jieba)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![AI Compatible](https://img.shields.io/badge/AI-OpenAI%20Compatible-purple)](https://platform.openai.com/)

> 基于 **Requests + BeautifulSoup + Jieba + LLM** 的每日经济新闻采集与分析系统 · 数据源: 人民网财经频道 + 中国经济网 · 输出: 6 维度结构化热点报告 (Markdown)

---

## ⚠️ 第一步:配置 AI API Key(必读)

本项目使用大语言模型(LLM)做政策与产业解读,**必须配置 API Key 才能生成完整报告**。

### 快速配置(2 分钟)

```bash
# 1. 复制模板
cp .env.example .env

# 2. 编辑 .env,填入你的 Key
# AI_API_KEY=sk-your-real-key-here
```

### 推荐 Provider(国内可用 + 便宜)

| Provider | 模型 | Base URL | 价格 | 备注 |
|---|---|---|---|---|
| **DeepSeek** ⭐推荐 | `deepseek-chat` | `https://api.deepseek.com/v1` | ¥1/百万 token | 国内访问,中文优秀 |
| 通义千问 Qwen | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 中 | 阿里云,稳定 |
| OpenAI | `gpt-4o-mini` | `https://api.openai.com/v1` | 中 | 需代理 |
| Claude | `claude-3-5-sonnet` | (使用 anthropic SDK) | 高 | 需代理 |

`.env` 文件已自动被 `.gitignore` 排除,**不会被提交到 GitHub**。

### 不配置 Key 也能跑

```bash
# 跳过 AI,降级为纯 NLP 报告(关键词 + 词频)
python main.py --skip-ai
```

详见下方 [💼 AI 接口选择](#-ai-接口选择)。

---

## 📑 目录

- [🎯 项目背景](#-项目背景)
- [📰 报告样例](#-报告样例)
- [🛠️ 技术栈](#️-技术栈)
- [📁 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)
- [🔬 分析流程](#-分析流程)
- [🧩 核心模块](#-核心模块)
- [💼 AI 接口选择](#-ai-接口选择)
- [📈 输出示例](#-输出示例)
- [❓ 常见问题](#-常见问题)
- [📝 License](#-license)

---

## 🎯 项目背景

经济新闻是观察政策风向和产业趋势的"窗口"。**《人民日报》** 作为党中央机关报,其经济新闻报道具有最高权威性与风向标意义。

本项目每天自动:

1. **采集** 人民网财经频道最近 N 天的经济新闻(每天约 11 篇置顶,通过 `--days N` 控制回溯天数,默认 7 天)
2. **分析** 通过 jieba 分词 + TF-IDF/TextRank 提取主题词
3. **解读** 通过 LLM(Claude/OpenAI/DeepSeek/Qwen)提炼政策与产业洞察
4. **归档** 输出 6 维度结构化报告,提交至 GitHub 形成历史时间序列

> ⚠️ **数据源特性**: 人民网财经主列表单日仅展示 ~11 篇置顶新闻(其中当天 9 篇 + 前几天 2 篇),且 `index1-5` 内容完全相同。因此本项目设计为**多日回溯**(默认 7 天)以保证日报内容丰富度,而非严格"前一天"。可通过 `--days N` 自定义。

**业务价值**:
- **投资者**: 每周 5 分钟速览宏观政策与产业风向
- **研究者**: 长期归档形成中国政策话语语料库
- **求职者**: 本项目展示 **数据采集 → NLP 处理 → LLM 工程化 → 报告自动化** 完整能力链

---

## 📰 报告样例

> 自动生成的每日报告示例(详见 `reports/2026-06-11.md`):

```markdown
# 人民日报经济新闻日报 · 2026-06-11

> 自动生成 · 基于 47 篇昨日经济新闻

## 📌 核心速览
昨日经济新闻聚焦"新质生产力"与"扩大内需"两大主线,涉及...

## 🔥 昨日主题词
| 排名 | 主题词 | 权重 |
|---|---|---|
| 1   | 新质生产力 | 0.087 |
| 2   | 扩大内需 | 0.076 |
| 3   | 数字经济 | 0.068 |
...

## 🌬️ 政策风向
🟢 **温和扩张** — 关键词: 降准预期、消费券、设备更新
```

---

## 🛠️ 技术栈

| 类别 | 工具 | 用途 |
|---|---|---|
| **爬虫** | requests + BeautifulSoup + lxml | HTML 采集与解析 |
| **限速** | time.sleep + retry | 反爬友好 |
| **中文 NLP** | jieba + scikit-learn | 分词 / TF-IDF / TextRank |
| **可视化** | wordcloud + matplotlib | 词云图生成 |
| **AI 推理** | openai SDK (兼容协议) | DeepSeek / Qwen / OpenAI / Claude |
| **报告** | Jinja2 | Markdown 模板渲染 |
| **配置** | python-dotenv | .env 管理 API Key |
| **打包** | argparse | CLI 入口 |

---

## 📁 项目结构

```
people-daily-economy-daily/
├── README.md                    # 项目门面
├── LICENSE                      # MIT
├── .gitignore
├── .env.example                 # AI Key 模板(必须自填)
├── requirements.txt
├── main.py                      # 一键 CLI 入口
├── push_to_github.ps1           # Windows 一键推送
├── GITHUB_GUIDE.md              # 部署与美化指南
├── src/
│   ├── config.py                # 配置加载 + 产业词典
│   ├── scraper/                 # 爬虫层
│   │   ├── fetcher.py           # requests 会话与限速
│   │   ├── list_parser.py       # 列表页解析
│   │   ├── article_parser.py    # 详情页解析
│   │   └── pipeline.py          # 串起列表→详情
│   ├── nlp/                     # 中文 NLP 层
│   │   ├── tokenizer.py         # jieba + 停用词
│   │   ├── keywords.py          # TF-IDF + TextRank
│   │   ├── stats.py             # 词频 + 产业匹配
│   │   └── wordcloud_gen.py     # 词云
│   ├── ai/                      # AI 推理层
│   │   ├── client.py            # OpenAI 兼容客户端
│   │   ├── prompts.py           # 6 个分析 prompt
│   │   └── analyzer.py          # 串接 6 个任务
│   ├── report/                  # 报告层
│   │   ├── template.md.j2       # Jinja2 模板
│   │   ├── renderer.py          # 渲染 + 写入
│   │   └── archiver.py          # JSON 归档
│   └── utils/
│       ├── logger.py            # 统一日志
│       └── date_utils.py        # 日期计算
├── data/
│   ├── raw/                     # 每日 JSON 归档
│   └── processed/               # 中间产物(可 gitignore)
├── reports/                     # 每日 Markdown 报告
├── images/                      # 词云图
├── notebooks/                   # 4 个分析 notebook
└── scripts/
    └── verify_selectors.py      # 实爬验证 CSS 选择器
```

---

## 🚀 快速开始

### 1. 克隆与安装

```bash
git clone https://github.com/jiayangrui05160031-cmyk/people-daily-economy-daily.git
cd people-daily-economy-daily
pip install -r requirements.txt
```

### 2. 配置 AI Key

```bash
cp .env.example .env
# 编辑 .env,填入 AI_API_KEY
```

支持的 provider: `openai` / `deepseek` / `qwen` / `custom`,详见 [💼 AI 接口选择](#-ai-接口选择)。

### 3. 运行

```bash
# 全流程:爬取 → NLP → AI → 报告
python main.py

# 指定日期 + 自定义回溯天数(默认 7)
python main.py --date 2026-06-11 --days 14

# 跳过爬取(用已有 JSON 重跑 AI/报告)
python main.py --skip-scrape

# 不调 LLM,降级为纯 NLP 报告(免 API key)
python main.py --skip-ai

# 调试日志
python main.py --debug
```

### 4. 查看报告

报告输出至 `reports/{日期}.md`,词云图在 `images/wordcloud_{日期}.png`,原始数据 JSON 在 `data/raw/{日期}.json`。

---

## 🔬 分析流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ STEP 1:爬取 │ -> │ STEP 2:NLP  │ -> │ STEP 3:AI   │
│ Requests +  │    │ Jieba +     │    │ LLM 6 个    │
│ BeautifulSoup│    │ TF-IDF/     │    │ Prompt      │
│ 抓前一天    │    │ TextRank    │    │ 任务        │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 最终报告    │ <- │ STEP 4:渲染 │ <- │ Analysis    │
│ reports/    │    │ Jinja2 +    │    │ Result      │
│ {date}.md   │    │ Markdown    │    │ (JSON)      │
└─────────────┘    └─────────────┘    └─────────────┘
```

**关键方法论**:
- **采集去重**:按文章 ID 去重,按 URL 路径日期过滤"前一天"
- **关键词融合**:TF-IDF(全局重要性) + TextRank(图排序)双路融合
- **LLM 提示工程**:6 个独立任务,每个返回 JSON Schema,失败降级到规则匹配
- **报告模板化**:Jinja2 渲染,可换肤

---

## 🧩 核心模块

| 模块 | 入口 | 职责 |
|---|---|---|
| **爬虫** | `src.scraper.pipeline.fetch_previous_day_articles` | 抓取 → 解析 → 返回 Article 列表 |
| **NLP** | `src.nlp.stats.analyze` | 分词 / 关键词 / 词频 / 词云 |
| **AI** | `src.ai.analyzer.analyze_all` | 6 个 LLM 任务 → AnalysisResult |
| **报告** | `src.report.renderer.render` | 渲染 → 写入 reports/ |

每个模块都可独立调用,便于测试与扩展。

---

## 📈 输出示例

每日报告结构(详见 `reports/{date}.md`):

| 章节 | 内容 | 来源 |
|---|---|---|
| 📌 核心速览 | 200 字摘要 | LLM |
| 🔥 昨日主题词 | Top 10 关键词 + 权重 | NLP + LLM |
| 🌬️ 政策风向 | 扩张/收紧/中性 + 关键词 | LLM |
| 🏭 重点产业/行业 | 3-5 个产业 + 涉及文章数 | 规则 + LLM |
| 📜 重点政策出台 | 3-5 条政策 + 主要内容 + 来源 | LLM |
| 🔮 未来发展判断 | 200 字趋势预测 | LLM |
| 📰 原始文章列表 | 所有文章标题/URL/来源/时间 | 爬虫 |

---

## ❓ 常见问题

**Q1: 没有任何新闻被采集到?**
A: 检查网络、确认 `finance.people.com.cn` 可访问;运行 `python scripts/verify_selectors.py` 验证选择器。

**Q2: LLM 调用失败?**
A: 检查 `.env` 中 `AI_API_KEY` 是否正确;网络是否能访问 `AI_BASE_URL`;DeepSeek/Qwen 无需代理,OpenAI/Claude 需要。

**Q3: 词云图中文是方块?**
A: Windows 自带字体在 `C:\Windows\Fonts\msyh.ttc`,已在 `config.py` 默认配置。

**Q4: 节假日新闻太少,报告很空?**
A: 正常现象,会生成精简版报告。周末和节假日新闻量约为工作日 1/3。

**Q5: 每天只能抓到 11 篇文章?**
A: 人民网财经主列表单日仅展示 ~11 篇置顶(且 index1-5 内容完全相同)。本项目默认回溯 7 天合并分析,可通过 `--days 30` 进一步扩大回溯窗口。

**Q6: 如何定时每天自动跑?**
A: Windows 用任务计划程序,Linux/Mac 用 crontab,详见 GITHUB_GUIDE.md。

---

## 📝 License

本项目采用 [MIT License](LICENSE)。

---

## ✨ 致谢

- 数据源: [人民网财经频道](http://finance.people.com.cn/)
- 中文分词: [jieba](https://github.com/fxsjy/jieba)
- 致敬所有开源贡献者

---

<p align="center">如果这个项目对你有帮助,欢迎 ⭐ Star!</p>
