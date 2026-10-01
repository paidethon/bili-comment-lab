# syntax=docker/dockerfile:1
# Bilibili 评论实验室 — single-container build.
# Data (raw/normalized/db/reports) lives in the mounted /app/data volume.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MALLOC_ARENA_MAX=2 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# runtime deps only (no build tools needed — all wheels).
# PIP_INDEX_URL is a build arg so CN networks can point at a local mirror:
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
COPY requirements.txt .
RUN pip install -r requirements.txt

# vendored crawler (MIT) installs as bilibili_crawler
COPY vendor/ /app/vendor/
RUN pip install --no-deps /app/vendor/BilibiliCrawler

# application code
COPY bilicom/ /app/bilicom/
COPY prompts/ /app/prompts/
COPY webapp/ /app/webapp/

# non-root, writable data dir
RUN useradd -m -u 10001 lab && mkdir -p /app/data && chown -R lab:lab /app/data /app
USER lab

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=4).status==200 else 1)"

# one worker is intentional: crawl/analyze jobs serialize on a semaphore
CMD ["python", "-m", "uvicorn", "webapp.app:app", \
     "--host", "0.0.0.0", "--port", "8765", \
     "--no-access-log", "--timeout-keep-alive", "30"]
