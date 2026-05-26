FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal OS packages for common document parsing/runtime needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY .env.example ./.env.example

# Runtime defaults (override in ECS task definition/environment as needed).
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8080 \
    APP_RELOAD=false \
    DATA_DIR=/app/data \
    DB_PATH=/app/table_line_item.db

EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn app.main:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8080}"]
