FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==3.10.1

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
