FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY henet_kb ./henet_kb
RUN pip install .

RUN mkdir -p /app/data
EXPOSE 8000
CMD ["uvicorn", "henet_kb.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
