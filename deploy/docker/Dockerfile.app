FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，利用镜像层缓存
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[web]"

COPY samples ./samples
COPY docs ./docs

EXPOSE 8080

CMD ["uvicorn", "tracecell.web.app:app", "--host", "0.0.0.0", "--port", "8080"]
