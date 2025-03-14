#!/bin/bash
set -e

# Настраиваем DVC, если конфигурация отсутствует
if [ ! -f .dvc/config ]; then
    poetry run dvc remote add -d myremote s3://ml-bucket
    poetry run dvc remote modify myremote endpointurl http://minio:9000
    poetry run dvc remote modify myremote access_key_id admin
    poetry run dvc remote modify myremote secret_access_key admin123
fi

# Запускаем mlflow UI в фоне (если требуется, можно вынести в отдельный контейнер)
poetry run mlflow ui --host 0.0.0.0 --port 5002 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns &

# Запускаем FastAPI приложение
exec poetry run uvicorn new_app:app --host 0.0.0.0 --port 8000
