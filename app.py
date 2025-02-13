from models import MODEL_REGISTRY
from data_validation import Dataset
from health import check_system_health

from fastapi import FastAPI, HTTPException, Request, Query,Path
import uvicorn

from pydantic import ValidationError
import json
import os
import numpy as np

app = FastAPI()

@app.post(
    "/train/{model_type}",
    summary="Обучение модели",
    tags=["Training"],
    responses={
        200: {
            "description": "Успешное обучение модели",
            "content": {
                "application/json": {
                    "example": {
                        "accuracy": 0.92,
                        "model_prefix": "rf",
                        "model_name": "rf_20250213_042111.pkl"
                    }
                }
            }
        },
        422: {"description": "Ошибка валидации входных данных"},
        400: {"description": "Некорректный формат запроса или неподдерживаемый тип модели"}
    },
    openapi_extra={
        "parameters": [
            {
                "in": "query",
                "name": "n_estimators",
                "schema": {"type": "integer", "default": 100, "minimum": 1},
                "description": (
                    "Количество деревьев для RandomForest. Дефолт: 100. "
                    "Если используется другая модель, параметры смотрите в /train/available_models."
                ),
                "example": 100
            },
            {
                "in": "query",
                "name": "max_depth",
                "schema": {"type": "integer", "default": 10, "minimum": 1},
                "description": (
                    "Максимальная глубина дерева для RandomForest. Дефолт: 10. "
                    "Если используется другая модель, параметры смотрите в /train/available_models."
                ),
                "example": 10
            },
            {
                "in": "query",
                "name": "iterations",
                "schema": {"type": "integer", "minimum": 1},
                "description": (
                    "Количество итераций для CatBoost. Если используется RandomForest — игнорируется. "
                    "Для остальных моделей смотрите /train/available_models."
                ),
                "example": 1000
            },
            {
                "in": "query",
                "name": "depth",
                "schema": {"type": "integer", "minimum": 1},
                "description": (
                    "Глубина дерева для CatBoost. Если используется RandomForest — игнорируется. "
                    "Для остальных моделей смотрите /train/available_models."
                ),
                "example": 6
            }
        ],
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "X": [
                            {
                                "PassengerId": 1,
                                "Pclass": 3,
                                "Name": "Braund, Mr. Owen Harris",
                                "Sex": "male",
                                "Age": 22,
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "A/5 21171",
                                "Fare": 7.25,
                                "Cabin": "",
                                "Embarked": "S"
                            },
                            {
                                "PassengerId": 2,
                                "Pclass": 1,
                                "Name": "Cumings, Mrs. John Bradley (Florence Briggs)",
                                "Sex": "female",
                                "Age": 38,
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "PC 17599",
                                "Fare": 71.2833,
                                "Cabin": "C85",
                                "Embarked": "C"
                            }
                        ],
                        "y": {
                            "1": 0,
                            "2": 1
                        }
                    }
                }
            }
        }
    }
)
async def train_model_endpoint(
    model_type: str = Path(
        ...,
        description="Тип модели (например, rf, catboost). Доступные модели: " + ", ".join(MODEL_REGISTRY.keys()) + ".",
        example="rf"
    ),
    request: Request = None
):
    # Валидация доступности модели
    if model_type not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Тип модели '{model_type}' не поддерживается. Доступные модели: {', '.join(MODEL_REGISTRY.keys())}."
        )

    query_params = dict(request.query_params)
    handler_cls = MODEL_REGISTRY.get(model_type)
    try:
        validated_params = handler_cls.params_schema(**query_params)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка в параметрах модели", "details": e.errors()}
        )
    try:
        body = await request.json()
        dataset = Dataset(**body)
        df = dataset.to_dataframe()
        if dataset.y is None:
            raise HTTPException(status_code=422, detail="Отсутствует целевая переменная 'y' в данных.")
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка в данных", "details": e.errors()}
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON в теле запроса")

    model = handler_cls()
    result = model.train(df, validated_params.dict())
    model_prefix = result.get("model_prefix", "model")
    model_name = model.save_model(model_prefix)
    result["model_name"] = model_name

    return result

@app.post(
    "/predict",
    summary="Предсказание по модели",
    tags=["Prediction"],
    responses={
        422: {"description": "Ошибка валидации входных данных"},
        400: {"description": "Некорректный формат запроса или модель не найдена"}
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "X": [
                            {
                                "PassengerId": 1,
                                "Pclass": 3,
                                "Name": "Braund, Mr. Owen Harris",
                                "Sex": "male",
                                "Age": 22,
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "A/5 21171",
                                "Fare": 7.25,
                                "Cabin": "",
                                "Embarked": "S"
                            },
                            {
                                "PassengerId": 2,
                                "Pclass": 1,
                                "Name": "Cumings, Mrs. John Bradley (Florence Briggs)",
                                "Sex": "female",
                                "Age": 38,
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "PC 17599",
                                "Fare": 71.2833,
                                "Cabin": "C85",
                                "Embarked": "C"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def predict_endpoint(
    model_name: str = Query(..., description="Имя модели для предсказания", example="catboost_20250213_042111.pkl"),
    request: Request = None  # оставляем валидацию через Request
):
    try:
        json_data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON в теле запроса")

    # Удаляем 'y', если он присутствует
    if isinstance(json_data, dict) and "y" in json_data:
        json_data.pop("y")

    try:
        dataset = Dataset(**json_data)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка в данных", "details": e.errors()}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Ошибка обработки входных данных для предсказания.")

    model_type = model_name.split("_")[0]
    model_handler = MODEL_REGISTRY.get(model_type)
    if model_handler is None:
        raise HTTPException(
            status_code=400,
            detail=f"Тип модели '{model_type}' не поддерживается."
        )

    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{model_name}' не найдена."
        )

    model = model_handler.load_model(model_name)
    df = dataset.to_dataframe()
    predictions = model.predict(df)
    if not isinstance(predictions, list):
        predictions = list(predictions)

    converted_predictions = []
    for pred in predictions:
        if isinstance(pred, (np.integer,)):
            converted_predictions.append(int(pred))
        elif isinstance(pred, (np.floating,)):
            converted_predictions.append(float(pred))
        else:
            converted_predictions.append(pred)

    passenger_ids = [passenger.PassengerId for passenger in dataset.X]
    prediction_dict = {int(pid): pred for pid, pred in zip(passenger_ids, converted_predictions)}

    return {"predictions": prediction_dict}


@app.post(
    "/predict",
    summary="Предсказание по модели",
    tags=["Prediction"],
    responses={
        200: {
            "description": "Успешное предсказание",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": {
                            "1": 0,
                            "2": 1
                        }
                    }
                }
            }
        },
        422: {"description": "Ошибка валидации входных данных"},
        400: {"description": "Некорректный формат запроса или модель не найдена"}
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "X": [
                            {
                                "PassengerId": 1,
                                "Pclass": 3,
                                "Name": "Braund, Mr. Owen Harris",
                                "Sex": "male",
                                "Age": 22,
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "A/5 21171",
                                "Fare": 7.25,
                                "Cabin": "",
                                "Embarked": "S"
                            },
                            {
                                "PassengerId": 2,
                                "Pclass": 1,
                                "Name": "Cumings, Mrs. John Bradley (Florence Briggs)",
                                "Sex": "female",
                                "Age": 38,
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "PC 17599",
                                "Fare": 71.2833,
                                "Cabin": "C85",
                                "Embarked": "C"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def predict_endpoint(
    model_name: str = Query(..., description="Имя модели для предсказания", example="catboost_20250213_042111.pkl"),
    request: Request = None
):
    try:
        json_data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON в теле запроса")

    if isinstance(json_data, dict) and "y" in json_data:
        json_data.pop("y")

    try:
        dataset = Dataset(**json_data)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка в данных", "details": e.errors()}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Ошибка обработки входных данных для предсказания.")

    model_type = model_name.split("_")[0]
    model_handler = MODEL_REGISTRY.get(model_type)
    if model_handler is None:
        raise HTTPException(
            status_code=400,
            detail=f"Тип модели '{model_type}' не поддерживается."
        )

    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{model_name}' не найдена."
        )

    model = model_handler.load_model(model_name)
    df = dataset.to_dataframe()
    predictions = model.predict(df)
    if not isinstance(predictions, list):
        predictions = list(predictions)

    converted_predictions = []
    for pred in predictions:
        if isinstance(pred, (np.integer,)):
            converted_predictions.append(int(pred))
        elif isinstance(pred, (np.floating,)):
            converted_predictions.append(float(pred))
        else:
            converted_predictions.append(pred)

    passenger_ids = [passenger.PassengerId for passenger in dataset.X]
    prediction_dict = {int(pid): pred for pid, pred in zip(passenger_ids, converted_predictions)}

    return {"predictions": prediction_dict}


@app.post(
    "/retrain",
    summary="Переобучение модели",
    tags=["Manage"],
    responses={
        200: {
            "description": "Успешное переобучение модели",
            "content": {
                "application/json": {
                    "example": {
                        "model_name": "rf_20250213_155448.pkl",
                        "accuracy": 0.93
                    }
                }
            }
        },
        422: {"description": "Ошибка валидации входных данных"},
        400: {"description": "Некорректный формат запроса или модель не поддерживается"},
        404: {"description": "Модель не найдена"},
        500: {"description": "Ошибка при загрузке, переобучении или сохранении модели"}
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "X": [
                            {
                                "PassengerId": 1,
                                "Pclass": 3,
                                "Name": "Braund, Mr. Owen Harris",
                                "Sex": "male",
                                "Age": 22,
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "A/5 21171",
                                "Fare": 7.25,
                                "Cabin": "",
                                "Embarked": "S"
                            },
                            {
                                "PassengerId": 2,
                                "Pclass": 1,
                                "Name": "Cumings, Mrs. John Bradley (Florence Briggs)",
                                "Sex": "female",
                                "Age": 38,
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "PC 17599",
                                "Fare": 71.2833,
                                "Cabin": "C85",
                                "Embarked": "C"
                            }
                        ],
                        "y": {
                            "1": 0,
                            "2": 1
                        }
                    }
                }
            }
        }
    }
)
async def retrain_model_endpoint(model_name: str = Query(..., description="Имя модели для переобучения",
                                                         example="rf_20250213_155448.pkl"), request: Request = None):
    """
    Эндпоинт для переобучения уже обученной модели.
    Имя модели передаётся через query-параметр `model_name`,
    новые данные — в теле запроса.
    Для переобучения используются сохранённые параметры модели.
    """
    # Валидация входных данных (ожидается JSON с данными для обучения)
    try:
        body = await request.json()
        dataset = Dataset(**body)
        df = dataset.to_dataframe()
        if dataset.y is None:
            raise HTTPException(status_code=422, detail="Отсутствует целевая переменная 'y' в данных.")
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка в данных", "details": e.errors()}
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON в теле запроса")

    # Определяем тип модели по префиксу в имени (например, 'rf' или 'catboost')
    try:
        model_type = model_name.split("_")[0]
    except Exception:
        raise HTTPException(status_code=400, detail="Некорректный формат имени модели.")

    handler_cls = MODEL_REGISTRY.get(model_type)
    if handler_cls is None:
        raise HTTPException(
            status_code=400,
            detail=f"Тип модели '{model_type}' не поддерживается."
        )

    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Модель '{model_name}' не найдена.")

    # Загружаем существующую модель
    try:
        model = handler_cls.load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке модели: {str(e)}")

    # Проверяем наличие сохранённых обучающих параметров
    if not hasattr(model, "trained_params") or model.trained_params is None:
        raise HTTPException(status_code=500, detail="Параметры модели не найдены для переобучения.")

    # Переобучаем модель на новых данных, используя сохранённые параметры
    try:
        result = model.train(df, model.trained_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при переобучении модели: {str(e)}")

    # Сохраняем переобученную модель (она будет сохранена с новым именем, основанным на префиксе)
    try:
        new_model_name = model.save_model(model_prefix=model_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении модели: {str(e)}")

    result["model_name"] = new_model_name
    return result


@app.get(
    "/train/available_models",
    summary="Получение списка доступных моделей",
    tags=["Training"],
    responses={
        200: {
            "description": "Список доступных моделей с описанием и схемой параметров",
            "content": {
                "application/json": {
                    "example": {
                      "rf": {
                        "description": "Модель Random Forest",
                        "params_schema": {
                          "properties": {
                            "n_estimators": {
                              "anyOf": [
                                {
                                  "exclusiveMinimum": 0,
                                  "type": "integer"
                                },
                                {
                                  "type": "null"
                                }
                              ],
                              "default": 100,
                              "description": "Количество деревьев",
                              "title": "N Estimators"
                            },
                            "max_depth": {
                              "anyOf": [
                                {
                                  "exclusiveMinimum": 0,
                                  "type": "integer"
                                },
                                {
                                  "type": "null"
                                }
                              ],
                              "default": 10,
                              "description": "Максимальная глубина дерева",
                              "title": "Max Depth"
                            },
                            "random_state": {
                              "default": 42,
                              "description": "Сид генератора случайных чисел",
                              "title": "Random State",
                              "type": "integer"
                            }
                          },
                          "title": "RandomForestParams",
                          "type": "object"
                        }
                      }
                    }

                }
            }
        }
    }
)
async def get_available_models():
    """
    Эндпоинт для получения списка доступных моделей, их описания и схемы параметров.
    Возвращает словарь, где ключ — это имя модели,
    а значение содержит описание и параметры, которые можно передать при обучении.
    """
    available = {}
    for model_type, handler_cls in MODEL_REGISTRY.items():
        available[model_type] = {
            "description": handler_cls.__doc__ or "Нет описания",
            "params_schema": handler_cls.params_schema.schema()
        }
    return available


@app.get(
    "/trained_models",
    summary="Получение списка обученных моделей",
    tags=["Training"],
    responses={
        200: {
            "description": "Список моделей успешно получен",
            "content": {
                "application/json": {
                    "example": {
                        "models": [
                            "rf_20250213_042111.pkl",
                            "catboost_20250213_042111.pkl"
                        ]
                    }
                }
            }
        },
        404: {"description": "Папка моделей не найдена"},
        500: {"description": "Ошибка получения списка моделей"}
    }
)
async def list_models():
    """
    Эндпоинт для получения списка всех обученных моделей из папки models.
    """
    models_dir = 'models'
    if not os.path.exists(models_dir):
        raise HTTPException(status_code=404, detail="Папка моделей не найдена.")

    try:
        # Получаем только файлы (если в папке могут быть и другие папки, их можно исключить)
        models = [
            model for model in os.listdir(models_dir)
            if os.path.isfile(os.path.join(models_dir, model))
        ]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Ошибка получения списка моделей.")


@app.delete(
    "/delete_model",
    summary="Удаление модели",
    tags=["Manage"],
    responses={
        200: {
            "description": "Модель успешно удалена",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Модель 'rf_20250213_042111.pkl' успешно удалена."
                    }
                }
            }
        },
        404: {"description": "Модель не найдена"},
        500: {"description": "Ошибка при удалении модели"}
    },
    openapi_extra={
        "parameters": [
            {
                "in": "query",
                "name": "model_name",
                "required": True,
                "schema": {"type": "string"},
                "description": "Имя модели для удаления. Пример: rf_20250213_042111.pkl",
                "example": "rf_20250213_042111.pkl"
            }
        ]
    }
)
async def delete_model_endpoint(model_name: str):
    """
    Эндпоинт для удаления файла модели.
    Принимает model_name в query-параметре.
    """
    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Модель '{model_name}' не найдена.")

    try:
        os.remove(model_path)
        return {"detail": f"Модель '{model_name}' успешно удалена."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при удалении модели: {str(e)}")


@app.get(
    "/health",
    summary="Проверка состояния системы",
    tags=["Health"],
    responses={
        200: {
            "description": "Состояние системы успешно получено",
            "content": {
                "application/json": {
                    "example": {
                        "health": {
                            "cpu": "5%",
                            "memory": "60%",
                            "disk": "80%"
                        }
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Эндпоинт для проверки состояния системы.
    Возвращает информацию о загрузке CPU, памяти и дискового пространства.
    """
    health_data = check_system_health()
    return {"health": health_data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)