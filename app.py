from models import MODEL_REGISTRY

from fastapi import FastAPI, HTTPException, Request
import uvicorn

from pydantic import ValidationError
import json
import os
from data_validation import Dataset

app = FastAPI()


@app.post("/train/{model_type}")
async def train_model_endpoint(model_type: str, request: Request):
    query_params = dict(request.query_params)
    handler_cls = MODEL_REGISTRY.get(model_type)

    if handler_cls is None:
        raise HTTPException(status_code=400, detail=f"Тип модели '{model_type}' не поддерживается.")

    try:
        validated_params = handler_cls.params_schema(**query_params)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"error": "Ошибка в параметрах модели", "details": e.errors()})

    try:
        body = await request.json()
        dataset = Dataset(**body)  # Валидация данных
        df = dataset.to_dataframe()

        # Проверка наличия y
        if dataset.y is None:
            raise HTTPException(status_code=422, detail="Отсутствует целевая переменная 'y' в данных.")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"error": "Ошибка в данных", "details": e.errors()})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON в теле запроса")

    model = handler_cls()

    result = model.train(df, validated_params.dict())

    model_prefix = result.get("model_prefix", "model")
    model_name = model.save_model(model_prefix)
    result["model_name"] = model_name

    return result

@app.post("/train/{model_type}")
async def train_model_endpoint(model_type: str, request: Request):
    query_params = dict(request.query_params)
    handler_cls = MODEL_REGISTRY.get(model_type)

    if handler_cls is None:
        raise HTTPException(status_code=400, detail=f"Тип модели '{model_type}' не поддерживается.")

    try:
        validated_params = handler_cls.params_schema(**query_params)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"error": "Ошибка в параметрах модели", "details": e.errors()})

    try:
        body = await request.json()
        dataset = Dataset(**body)  # Валидация данных
        df = dataset.to_dataframe()

        # Проверка наличия y
        if dataset.y is None:
            raise HTTPException(status_code=422, detail="Отсутствует целевая переменная 'y' в данных.")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"error": "Ошибка в данных", "details": e.errors()})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON в теле запроса")

    model = handler_cls()

    result = model.train(df, validated_params.dict())

    model_prefix = result.get("model_prefix", "model")
    model_name = model.save_model(model_prefix)
    result["model_name"] = model_name

    return result

@app.get("/train/available_models")
async def get_available_models():
    available = {}
    for model_type, handler_cls in MODEL_REGISTRY.items():
        available[model_type] = {
            "description": handler_cls.__doc__ or "Нет описания",
            "params_schema": handler_cls.params_schema.schema()
        }
    return available

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.post("/predict")
async def predict_endpoint(model_name: str, request: Request):
    """
    Эндпоинт для предсказания.
    Принимает model_name в query-параметре и список записей в теле запроса.
    """
    try:
        json_data = await request.json()  # Ожидается словарь с данными
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON в теле запроса")

    # Удаляем 'y', если он присутствует, т.к. для тестирования целевая переменная не нужна
    if isinstance(json_data, dict) and 'y' in json_data:
        json_data.pop('y')

    try:
        dataset = Dataset(**json_data)
        df = dataset.to_dataframe()
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка в данных", "details": e.errors()}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Ошибка обработки входных данных для предсказания.")

    # Определяем тип модели по префиксу в имени файла (например, "rf" или "lr")
    model_type = model_name.split("_")[0]
    model_handler = MODEL_REGISTRY.get(model_type)
    if model_handler is None:
        raise HTTPException(
            status_code=400,
            detail=f"Тип модели '{model_type}' не поддерживается."
        )

    model_path = f'models/{model_name}'
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{model_name}' не найдена."
        )

    model = model_handler.load_model(model_name)
    predictions = model.predict(df).tolist()

    return {"predictions": predictions}
