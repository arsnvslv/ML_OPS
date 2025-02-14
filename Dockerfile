FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git curl gcc build-essential libssl-dev dos2unix \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=2.0.1
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

COPY pyproject.toml poetry.lock /app/
RUN poetry config virtualenvs.create false
RUN poetry install

# Копируем исходный код приложения
COPY . .

# Копируем и конвертируем entrypoint.sh
COPY entrypoint.sh /app/
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 8000
EXPOSE 5002

CMD ["/app/entrypoint.sh"]
