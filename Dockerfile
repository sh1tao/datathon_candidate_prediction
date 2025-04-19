FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY model/ ./model

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "8000"]