## Данные

### Структура данных
- **Формат:** JSON (конвертирован из CSV)
- **Train/Eval:** Объект с ключами:
  - `X`: список словарей, где ключи – фичи, а значения – их значения.
  - `y`: словарь, где ключи – `PassengerId`, а значения – целевая переменная (например, `Survived`).

Подробнее см. в файле `titanic/splitting.py`.

### Валидация
- **Инструмент:** Pydantic
- **Логика:** Для каждого пассажира создаётся объект `Passenger` (валидация типов), а затем весь датасет (`X` и `y`) проверяется на согласованность. Подробнее в `data_validation.py`.

---

## Модели

- **Препроцессинг:** Все модели используют единый препроцессинг (imputer, scaler, one-hot encoding), реализованный в базовом классе `BaseMlModel`.
- **Методы:** Каждая модель реализует методы `train` и `predict`, а также абстрактные методы `_train`, `_predict`, `save_model` и `load_model`.
- **Регистрация:** Модели регистрируются в `MODEL_REGISTRY` (см. `models.py`).

---

## FastAPI

### Эндпоинты

- **Обучение модели:**  
  `POST /train/{model_type}`  
  Пример:
  ```bash
  curl -X POST "http://127.0.0.1:8000/train/catboost?iterations=1000&depth=6" \
       -H "Content-Type: application/json" \
       -d @titanic/train_data.json
```

**Список обученных моделей:**  
`GET /trained_models`  
Пример:
```bash 
curl -X GET "http://127.0.0.1:8000/trained_models"
```

**Предсказание:**  
`POST /predict`  
Пример:
```bash 
curl -X POST "http://127.0.0.1:8000/predict?model_name=catboost_20250213_042111.pkl" \
     -H "Content-Type: application/json" \
     -d @titanic/test_data.json

```

**Переобучение:**  
`POST /retrain`  
Пример:
```bash 
curl -X POST "http://127.0.0.1:8000/retrain?model_name=catboost_20250213_155448.pkl" \
     -H "Content-Type: application/json" \
     -d @titanic/eval_data.json
```

**Удаление модели:**  
`DELETE /delete_model`  
Пример:
```bash
curl -X DELETE "http://127.0.0.1:8000/delete_model?model_name=catboost_20250213_155448.pkl"
```

**Доступные модели:**  
`GET /train/available_models`  
Пример:
```bash
curl -X GET "http://127.0.0.1:8000/train/available_models"
```

**Health-check:**  
`GET /health`  
Пример:
```
curl -X GET "http://127.0.0.1:8000/health"
```

### Запуск FastAPI

Запуск:
```bash
poetry run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```


### Запуск gRPC

1. gRPC-сервис предоставляет аналогичный функционал:

- **TrainModel:** Обучение модели с переданными данными и гиперпараметрами.
- **Predict:** Предсказание с использованием сохранённой модели.
- **Retrain:** Переобучение модели на новых данных.
- **ListTrainedModels:** Получение списка обученных моделей.
- **DeleteModel:** Удаление модели.
- **HealthCheck:** Проверка состояния системы.
- **ListAvailableModels:** Получение списка доступных моделей с описанием и схемой гиперпараметров.

 Запустите gRPC сервер:
```bash 
poetry run python grpc_server.py
```

После запуски сервера запускаем скрипт с клиентом (внутри скрипта просто набор вызовов каждого эндпоинта для проверки)
```bash 
poetry run python client.py
```

## Логирование

- **FastAPI:** Логи записываются в файл `fastapi_app.log` и выводятся в консоль.
- **gRPC:** Логирование осуществляется с использованием стандартного модуля `logging`

## Streamlit

В dashboard.py лежит код для простого EDA для titanic/titanic.csv  
Для запуска выполняем 
```bash
poetry run streamlit run dashboard.py
```
