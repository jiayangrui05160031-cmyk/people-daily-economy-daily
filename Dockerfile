# ============================================================
# Dockerfile — 宏观经济智能分析平台 v9
# 多阶段: builder (装 deps) + runtime (minimal)
# 基础: python:3.11-slim, 镜像 ~280MB
# ============================================================
FROM python:3.11-slim AS builder

# 防止 Python 写 .pyc, 强制 stdout 不缓冲
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# 系统依赖 (lxml + wordcloud + matplotlib 需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        libjpeg-dev \
        zlib1g-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ============================================================
# Runtime 阶段 — 干净的运行时镜像
# ============================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=Asia/Shanghai \
    LANG=zh_CN.UTF-8 \
    LC_ALL=C.UTF-8

# 非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 必要的 runtime 库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxml2 \
        libxslt1.1 \
        libjpeg62-turbo \
        zlib1g \
        tini \
        tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 复制 site-packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目代码
COPY --chown=appuser:appuser . /app

# 创建数据 / 日志目录并赋权
RUN mkdir -p /app/data/raw /app/data/processed /app/data/historical /app/data/cache \
             /app/reports /app/dashboard /app/images /app/logs \
    && chown -R appuser:appuser /app

USER appuser

# 健康检查 (FastAPI /health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

EXPOSE 8000

# 入口脚本
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-X", "utf8", "-m", "src.api.server", "--host", "0.0.0.0", "--port", "8000"]
