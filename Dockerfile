FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements.txt .
COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY data/processed ./data/processed

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]