"""
FastAPI-сервис для работы с моделями машинного обучения.

Эндпоинты позволяют:
- обучать модель (/train/{model_type}),
- делать предсказание (/predict),
- переобучать модель (/retrain),
- получать список доступных моделей (/train/available_models),
- получать список обученных моделей (/trained_models),
- удалять модель (/delete_model),
- проверять состояние системы (/health).

Логирование важных событий выполняется с помощью стандартного модуля logging.
Логи записываются в файл 'fastapi_app.log' (а также выводятся в консоль).
"""

import os
import json
import logging
import numpy as np

from fastapi import FastAPI, HTTPException, Request, Query, Path
from pydantic import ValidationError
import uvicorn

from models import MODEL_REGISTRY
from data_validation import Dataset
from health import check_system_health

# Настройка логирования: записи в файл и вывод в консоль.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fastapi_app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

app = FastAPI(
    title="ML Models Service API",
    description="API для обучения, предсказания, переобучения и управления моделями машинного обучения.",
    version="1.0.0"
)


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
                                "Age": "22",
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "A/5 21171",
                                "Fare": "7.25",
                                "Cabin": "",
                                "Embarked": "S"
                            },
                            {
                                "PassengerId": 2,
                                "Pclass": 1,
                                "Name": "Cumings, Mrs. John Bradley (Florence Briggs)",
                                "Sex": "female",
                                "Age": "38",
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "PC 17599",
                                "Fare": "71.2833",
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
    """
    Эндпоинт для обучения модели.

    Параметры для обучения передаются через query-параметры,
    а обучающие данные – в теле запроса (JSON).
    """
    logging.info("Получен запрос на обучение модели типа '%s'", model_type)

    if model_type not in MODEL_REGISTRY:
        err = f"Тип модели '{model_type}' не поддерживается. Доступные модели: {', '.join(MODEL_REGISTRY.keys())}."
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    query_params = dict(request.query_params)
    handler_cls = MODEL_REGISTRY.get(model_type)
    try:
        validated_params = handler_cls.params_schema(**query_params)
        logging.info("Параметры модели успешно валидированы: %s", validated_params.dict())
    except ValidationError as e:
        logging.error("Ошибка валидации параметров модели: %s", e.errors())
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка в параметрах модели", "details": e.errors()}
        )

    try:
        body = await request.json()
        dataset = Dataset(**body)
        df = dataset.to_dataframe()
        if dataset.y is None:
            err = "Отсутствует целевая переменная 'y' в данных."
            logging.error(err)
            raise HTTPException(status_code=422, detail=err)
        logging.info("Данные успешно получены и валидированы для обучения.")
    except ValidationError as e:
        logging.error("Ошибка валидации данных: %s", e.errors())
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка в данных", "details": e.errors()}
        )
    except json.JSONDecodeError:
        err = "Некорректный JSON в теле запроса."
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    model = handler_cls()
    try:
        result = model.train(df, validated_params.dict())
        model_prefix = result.get("model_prefix", "model")
        model_name = model.save_model(model_prefix)
        result["model_name"] = model_name
        logging.info("Модель успешно обучена и сохранена под именем: %s", model_name)
    except Exception as e:
        err = f"Ошибка при обучении модели: {str(e)}"
        logging.error(err)
        raise HTTPException(status_code=500, detail=err)

    return result


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
                                "Age": "22",
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "A/5 21171",
                                "Fare": "7.25",
                                "Cabin": "",
                                "Embarked": "S"
                            },
                            {
                                "PassengerId": 2,
                                "Pclass": 1,
                                "Name": "Cumings, Mrs. John Bradley (Florence Briggs)",
                                "Sex": "female",
                                "Age": "38",
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "PC 17599",
                                "Fare": "71.2833",
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
    """
    Эндпоинт для предсказания.

    Принимает имя модели (query-параметр) и данные для предсказания в теле запроса.
    Если в данных присутствует ключ "y", он будет удалён.
    """
    logging.info("Получен запрос на предсказание для модели: %s", model_name)
    try:
        json_data = await request.json()
    except json.JSONDecodeError:
        err = "Некорректный JSON в теле запроса."
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    if isinstance(json_data, dict) and "y" in json_data:
        json_data.pop("y")
        logging.info("Ключ 'y' удалён из входных данных для предсказания.")

    try:
        dataset = Dataset(**json_data)
        logging.info("Данные для предсказания успешно валидированы.")
    except ValidationError as e:
        logging.error("Ошибка валидации данных для предсказания: %s", e.errors())
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка в данных", "details": e.errors()}
        )
    except Exception:
        err = "Ошибка обработки входных данных для предсказания."
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    try:
        model_type = model_name.split("_")[0]
    except Exception:
        err = "Некорректный формат имени модели."
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    model_handler = MODEL_REGISTRY.get(model_type)
    if model_handler is None:
        err = f"Тип модели '{model_type}' не поддерживается."
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        err = f"Модель '{model_name}' не найдена."
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    try:
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
        logging.info("Предсказание успешно выполнено для модели: %s", model_name)
    except Exception as e:
        err = f"Ошибка при предсказании: {str(e)}"
        logging.error(err)
        raise HTTPException(status_code=500, detail=err)

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
                                "Age": "22",
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "A/5 21171",
                                "Fare": "7.25",
                                "Cabin": "",
                                "Embarked": "S"
                            },
                            {
                                "PassengerId": 2,
                                "Pclass": 1,
                                "Name": "Cumings, Mrs. John Bradley (Florence Briggs)",
                                "Sex": "female",
                                "Age": "38",
                                "SibSp": 1,
                                "Parch": 0,
                                "Ticket": "PC 17599",
                                "Fare": "71.2833",
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
async def retrain_model_endpoint(
    model_name: str = Query(..., description="Имя модели для переобучения", example="rf_20250213_155448.pkl"),
    request: Request = None
):
    """
    Эндпоинт для переобучения модели.

    Имя модели передаётся через query-параметр, новые данные – в теле запроса.
    Переобучение проводится с использованием сохранённых параметров модели.
    """
    logging.info("Получен запрос на переобучение модели: %s", model_name)
    try:
        body = await request.json()
        dataset = Dataset(**body)
        df = dataset.to_dataframe()
        if dataset.y is None:
            err = "Отсутствует целевая переменная 'y' в данных."
            logging.error(err)
            raise HTTPException(status_code=422, detail=err)
        logging.info("Данные для переобучения успешно валидированы.")
    except ValidationError as e:
        logging.error("Ошибка валидации данных для переобучения: %s", e.errors())
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка в данных", "details": e.errors()}
        )
    except json.JSONDecodeError:
        err = "Некорректный JSON в теле запроса."
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    try:
        model_type = model_name.split("_")[0]
    except Exception:
        err = "Некорректный формат имени модели."
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    handler_cls = MODEL_REGISTRY.get(model_type)
    if handler_cls is None:
        err = f"Тип модели '{model_type}' не поддерживается."
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        err = f"Модель '{model_name}' не найдена."
        logging.error(err)
        raise HTTPException(status_code=404, detail=err)

    try:
        model = handler_cls.load_model(model_name)
        logging.info("Модель '%s' успешно загружена для переобучения.", model_name)
    except Exception as e:
        err = f"Ошибка при загрузке модели: {str(e)}"
        logging.error(err)
        raise HTTPException(status_code=500, detail=err)

    if not hasattr(model, "trained_params") or model.trained_params is None:
        err = "Параметры модели не найдены для переобучения."
        logging.error(err)
        raise HTTPException(status_code=500, detail=err)

    try:
        result = model.train(df, model.trained_params)
        new_model_name = model.save_model(model_prefix=model_type)
        result["model_name"] = new_model_name
        logging.info("Модель переобучена и сохранена под именем: %s", new_model_name)
    except Exception as e:
        err = f"Ошибка при переобучении модели: {str(e)}"
        logging.error(err)
        raise HTTPException(status_code=500, detail=err)

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
    Эндпоинт для получения списка доступных моделей с описанием и схемой параметров.
    """
    logging.info("Получен запрос на список доступных моделей.")
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
    logging.info("Получен запрос на список обученных моделей.")
    models_dir = 'models'
    if not os.path.exists(models_dir):
        err = "Папка моделей не найдена."
        logging.error(err)
        raise HTTPException(status_code=404, detail=err)
    try:
        models = [
            model for model in os.listdir(models_dir)
            if os.path.isfile(os.path.join(models_dir, model))
        ]
        logging.info("Найдено %d моделей.", len(models))
        return {"models": models}
    except Exception as e:
        err = "Ошибка получения списка моделей."
        logging.error("%s: %s", err, str(e))
        raise HTTPException(status_code=500, detail=err)


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
    }
)
async def delete_model_endpoint(model_name: str = Query(..., description="Имя модели для удаления", example="rf_20250213_042111.pkl")):
    """
    Эндпоинт для удаления модели.

    Принимает model_name в query-параметре.
    """
    logging.info("Получен запрос на удаление модели: %s", model_name)
    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        err = f"Модель '{model_name}' не найдена."
        logging.error(err)
        raise HTTPException(status_code=404, detail=err)

    try:
        os.remove(model_path)
        logging.info("Модель '%s' успешно удалена.", model_name)
        return {"detail": f"Модель '{model_name}' успешно удалена."}
    except Exception as e:
        err = f"Ошибка при удалении модели: {str(e)}"
        logging.error(err)
        raise HTTPException(status_code=500, detail=err)


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
    logging.info("Получен запрос HealthCheck.")
    health_data = check_system_health()
    logging.info("HealthCheck: %s", health_data)
    return {"health": health_data}


if __name__ == "__main__":
    logging.info("Запуск FastAPI-сервиса на порту 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)