# GitHub 部署与美化指南

> 5 分钟把 `people-daily-economy-daily` 推到 GitHub,并让它"看起来很专业"。

---

## 一、5 分钟推送到 GitHub

### 1. 前置条件

| 工具 | 用途 | 检查命令 |
|---|---|---|
| git | 版本控制 | `git --version` |
| GitHub 账号 | 托管 | https://github.com |

### 2. 在 GitHub 创建空仓库

1. 打开 https://github.com/new
2. **Repository name**: `people-daily-economy-daily`
3. **Description**: `每日人民日报经济新闻采集 + NLP + AI 分析报告`
4. **Public** (简历项目建议公开)
5. **不要勾选** "Add a README" / ".gitignore" / "license"(我们已经有了)
6. 点击 **Create repository**

### 3. 运行推送脚本

```powershell
cd D:\cluade_outpute\people-daily-economy-daily
.\push_to_github.ps1
```

按提示输入 GitHub 用户名即可。

### 4. 认证失败的两种解法

**方案 A - Personal Access Token (推荐)**:
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token,勾选 `repo`
3. 推送时 Username 填 GitHub 用户名,Password 填 Token

**方案 B - SSH**:
```powershell
ssh-keygen -t ed25519
# 把 ~/.ssh/id_ed25519.pub 内容加到 GitHub SSH keys
git remote set-url origin git@github.com:<user>/people-daily-economy-daily.git
git push -u origin main
```

---

## 二、让项目更好看的 5 个加分动作

### 1. About 区描述 + Topics

进入 repo → 右侧 ⚙️ **About** → 设置:

- **Description**: `每日人民日报经济新闻采集 + NLP + AI 分析的自动化报告系统`
- **Website**: 留空(或填你的博客)
- **Topics** (9 个推荐):
  ```
  python, requests, beautifulsoup, jieba, nlp, llm, 
  openai, news-analysis, china-policy
  ```

### 2. 上传示例报告截图

1. 跑一次 `python main.py` 生成 `reports/{昨天}.md`
2. 用 Typora / VSCode 打开,截图核心速览部分
3. 放到 `images/sample-report.png`
4. 在 README 替换示例段为 `<img src="images/sample-report.png">`

### 3. 添加 Shields.io 徽章(已自带)

README 已有 5 个标准徽章,可根据需要追加:
- ![GitHub stars](https://img.shields.io/github/stars/USER/REPO)
- ![GitHub forks](https://img.shields.io/github/forks/USER/REPO)
- ![GitHub last commit](https://img.shields.io/github/last-commit/USER/REPO)

### 4. 启用 GitHub Issues

Settings → General → Features → ✅ Issues

这样用户能提 bug,你也能展示"维护中"。

### 5. 添加 Releases

每发布一版(如 v1.0.0)打 tag:
```powershell
git tag -a v1.0.0 -m "首个稳定版"
git push origin v1.0.0
```

然后在 GitHub → Releases → Draft a new release → 选择 tag。

---

## 三、推荐项目名备选

如果你觉得 `people-daily-economy-daily` 太长,可以选:

| 英文名 | 含义 |
|---|---|
| `people-daily-economy-daily` | 完整描述(推荐,SEO 友好) |
| `pd-econ-daily` | 简短缩写 |
| `china-econ-news-digest` | 国际化定位 |
| `rmrb-econ-bot` | 人民日报缩写 + bot 风格 |

---

## 四、常见问题

| 问题 | 解答 |
|---|---|
| **中文路径报错** | 项目路径必须是英文,推荐 `D:\cluade_outpute\` |
| **PDF/截图太大** | 用 https://tinypng.com 压缩后再 commit |
| **依赖装不上** | 用 `pip install --user -r requirements.txt` 或换 venv |
| **GitHub 25MB 限制** | reports/*.md 已 gitignore;如要保留历史报告建议加 LFS |
| **API Key 误提交** | 立即在对应平台 revoke,然后从 git history 清除(`git filter-repo`) |

---

## 五、定时运行(可选进阶)

### Windows 任务计划

1. `Win + R` → `taskschd.msc`
2. 创建基本任务 → 触发器:每天 07:00
3. 操作:启动程序 `python`,参数 `D:\cluade_outpute\people-daily-economy-daily\main.py`
4. 完成后用 GitHub Action 自动 commit 报告(可后续添加)

### Linux crontab

```bash
0 7 * * * cd /path/to/people-daily-economy-daily && python main.py >> logs/cron.log 2>&1
```

---

## 六、简历呈现建议

在简历"项目经历"栏目,可这样写:

> **人民日报经济新闻日报系统** | Python · 数据采集 · NLP · LLM
> - 设计并实现每日自动化数据流水线,采集《人民日报》财经频道 30-80 篇经济新闻
> - 基于 jieba + TF-IDF/TextRank 提取主题词,通过 LLM(DeepSeek/Qwen)生成 6 维度政策分析报告
> - 报告自动归档至 GitHub,形成历史时间序列,辅助宏观研究决策
> - 技术栈:requests / BeautifulSoup / jieba / scikit-learn / OpenAI SDK / Jinja2

⭐ **核心加分点**: 完整闭环(采集→分析→AI→报告→归档)、真实数据源、可演示。