import os
import json
import logging
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query, Path
from pydantic import ValidationError
import numpy as np

from models import MODEL_REGISTRY
from data_validation import Dataset
from health import check_system_health

from s3_service import upload_to_s3, download_from_s3, create_bucket
from dvc_service import dvc_add_push, dvc_pull, run_dvc_command

import uvicorn

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fastapi_app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

app = FastAPI(
    title="ML Models Service API (with DVC + Minio)",
    description="API для обучения, предсказания, переобучения и управления моделями машинного обучения (с DVC/Minio).",
    version="1.0.0"
)

try:
    create_bucket()
except Exception as e:
    logging.error(f"Failed to create MinIO bucket: {e}")


def clean_datasets_folder():
    """
    Удаляем всё в папке datasets/ и заново её создаём пустой.
    """
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    if os.path.exists(datasets_dir):
        shutil.rmtree(datasets_dir)
    os.makedirs(datasets_dir, exist_ok=True)
    logging.info("Cleaned up the datasets folder.")


@app.post(
    "/upload-data",
    summary="Загрузка датасета в DVC/Minio",
    tags=["Data"],
    responses={
        200: {
            "description": "Датасет успешно загружен и заверсирован",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Dataset uploaded and versioned successfully",
                        "dvc_file": "dvc/train_data.json.dvc"
                    }
                }
            }
        },
        500: {"description": "Ошибка при загрузке датасета"},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "Файл датасета для загрузки (JSON, CSV и т.д.)",
                            }
                        },
                        "required": ["file"]
                    },
                    "encoding": {
                        "file": {
                            "contentType": "application/octet-stream"
                        }
                    }
                }
            }
        }
    }
)
async def upload_data_endpoint(file: UploadFile = File(...)):
    """
    Загружает датасет в DVC + Minio.

    **Шаги**:
    1. Принимает файл (UploadFile) в формате multipart.
    2. Сохраняет во временную папку `datasets/`.
    3. `dvc add + dvc push` -> файл уходит в Minio (S3 remote).
    4. .dvc файл тоже загружается в Minio по пути `dvc/<название>.dvc`.
    5. Очищаем `datasets/`.
    """
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    try:
        dataset_path = os.path.join(datasets_dir, file.filename)
        with open(dataset_path, "wb") as f:
            f.write(await file.read())
        logging.info(f"File '{file.filename}' saved locally at {dataset_path}")

        dvc_add_push(dataset_path)  # (внутри вызывает dvc add и dvc push)

        dvc_file_path = dataset_path + ".dvc"
        dvc_s3_key = f"dvc/{os.path.basename(dvc_file_path)}"
        upload_to_s3(dvc_file_path, dvc_s3_key)
        logging.info(f".dvc file uploaded to Minio: {dvc_s3_key}")

    except Exception as e:
        logging.error(f"Failed to upload dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload dataset")
    finally:
        clean_datasets_folder()

    return {
        "message": "Dataset uploaded and versioned successfully",
        "dvc_file": dvc_s3_key
    }


@app.post(
    "/train/{model_type}",
    summary="Обучение модели из DVC/Minio",
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
        422: {"description": "Ошибка валидации гиперпараметров или датасета"},
        400: {"description": "Некорректный формат запроса или неподдерживаемый тип модели"},
        500: {"description": "Ошибка при загрузке датасета или обучении модели"}
    },
    openapi_extra={
        "parameters": [
            {
                "in": "query",
                "name": "n_estimators",
                "schema": {"type": "integer", "default": 100, "minimum": 1},
                "description": "Количество деревьев для RandomForest. По умолчанию: 100."
            },
            {
                "in": "query",
                "name": "max_depth",
                "schema": {"type": "integer", "default": 10, "minimum": 1},
                "description": "Максимальная глубина дерева для RandomForest. По умолчанию: 10."
            },
            {
                "in": "query",
                "name": "random_state",
                "schema": {"type": "integer", "default": 42},
                "description": "Сид генератора случайных чисел."
            }
        ],
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "dvc_key": "dvc/train_data.json.dvc"
                    }
                }
            }
        }
    }
)
@app.post(
    "/train/{model_type}",
)
async def train_model_endpoint(
    model_type: str = Path(
        ...,
        description="Тип модели (например, rf, catboost).",
        example="rf"
    ),
    request: Request = None
):
    """
    Эндпоинт для обучения модели.

    **Шаги**:
    1) Получаем гиперпараметры из query (request.query_params).
    2) Из тела берем {"dvc_key": "..."} и подтягиваем датасет через DVC + Minio.
    3) Обучаем модель.
    4) Логируем гиперпараметры, метрику и модель (артефакт) в MLflow.
    """
    logging.info("Получен запрос на обучение модели типа '%s'", model_type)

    # 1) Проверка типа модели
    if model_type not in MODEL_REGISTRY:
        err = f"Тип модели '{model_type}' не поддерживается. Доступные: {', '.join(MODEL_REGISTRY.keys())}"
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    # 2) Читаем гиперпараметры из query
    query_params = dict(request.query_params)
    logging.info(f"Получены query-параметры: {query_params}")

    handler_cls = MODEL_REGISTRY[model_type]
    try:
        validated_params = handler_cls.params_schema(**query_params)
        logging.info("Валидированные гиперпараметры: %s", validated_params.dict())
    except ValidationError as e:
        logging.error("Ошибка валидации гиперпараметров: %s", e.errors())
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка в параметрах модели", "details": e.errors()}
        )

    # 3) Считываем тело -> {"dvc_key": "..."}
    try:
        body = await request.json()
        if "dvc_key" not in body:
            err = "В теле запроса отсутствует поле 'dvc_key'."
            logging.error(err)
            raise HTTPException(status_code=422, detail=err)
        dvc_key = body["dvc_key"]
        logging.info(f"dvc_key из тела: {dvc_key}")
    except json.JSONDecodeError:
        err = "Некорректный JSON в теле запроса."
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)
    except Exception as e:
        err = f"Ошибка при чтении тела запроса: {str(e)}"
        logging.error(err)
        raise HTTPException(status_code=400, detail=err)

    # 4) Подгружаем датасет через DVC
    try:
        clean_datasets_folder()
        local_dvc_file = os.path.join("datasets", os.path.basename(dvc_key))

        download_from_s3(dvc_key, local_dvc_file)
        logging.info(f".dvc файл скачан из Minio: {dvc_key} -> {local_dvc_file}")

        run_dvc_command(["dvc", "pull"])
        dataset_path = local_dvc_file.replace(".dvc", "")
        if not os.path.exists(dataset_path):
            err = f"После dvc pull не найден датасет {dataset_path}"
            logging.error(err)
            raise HTTPException(status_code=500, detail=err)

        with open(dataset_path, "r", encoding="utf-8") as f:
            data_content = json.load(f)

        dataset = Dataset(**data_content)
        df = dataset.to_dataframe()
        if dataset.y is None:
            err = "Отсутствует целевая переменная 'y' в датасете."
            logging.error(err)
            raise HTTPException(status_code=422, detail=err)

        logging.info(f"Датасет успешно загружен. {len(df)} строк.")

    except ValidationError as e:
        logging.error("Ошибка валидации датасета: %s", e.errors())
        raise HTTPException(
            status_code=422,
            detail={"error": "Ошибка датасета", "details": e.errors()}
        )
    except HTTPException:
        raise  # пробрасываем дальше
    except Exception as e:
        err = f"Ошибка при подгрузке датасета из DVC: {str(e)}"
        logging.error(err)
        raise HTTPException(status_code=500, detail=err)

    # 5) Обучаем модель и логируем в MLflow
    model = handler_cls()
    try:
        # Настройка MLflow
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("MyMLProject")  # Название эксперимента

        with mlflow.start_run(run_name=f"Train_{model_type}"):
            # Логируем гиперпараметры
            for param_name, param_value in validated_params.dict().items():
                mlflow.log_param(param_name, param_value)

            # Само обучение
            result = model.train(df, validated_params.dict())

            # Логируем метрику (если есть, напр. accuracy)
            if "accuracy" in result:
                mlflow.log_metric("accuracy", result["accuracy"])

            # Сохраняем кастомную модель (ваш метод):
            model_prefix = result.get("model_prefix", "model")
            model_name = model.save_model(model_prefix)  # например, "rf_20250615_123456.pkl"
            result["model_name"] = model_name
            logging.info("Модель обучена и сохранена как %s", model_name)

            # Логируем этот файл в MLflow как артефакт
            model_path = os.path.join("models", model_name)
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="custom_model")
                logging.info(f"Файл {model_path} залогирован в MLflow как артефакт.")
            else:
                logging.warning(f"Файл модели {model_path} не найден, не удалось логировать артефакт.")

        # end of with mlflow.start_run
    except Exception as e:
        err = f"Ошибка при обучении модели: {str(e)}"
        logging.error(err)
        raise HTTPException(status_code=500, detail=err)
    finally:
        clean_datasets_folder()

    return result



@app.post(
    "/retrain",
    summary="Переобучение модели",
    tags=["Training"],
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
        422: {"description": "Ошибка валидации данных или гиперпараметров"},
        400: {"description": "Некорректный формат запроса или модель не поддерживается"},
        404: {"description": "Модель не найдена"},
        500: {"description": "Ошибка при переобучении"}
    }
)
async def retrain_model_endpoint(
    model_name: str = Query(..., description="Имя уже обученной модели (файл .pkl)"),
    dvc_key: str = Query(..., description="Ключ к .dvc-файлу (новые данные)"),
    request: Request = None
):
    """
    Переобучаем существующую модель:
    1) Скачиваем .dvc-файл (dvc_key) для обновлённого датасета
    2) `dvc pull` -> локальный JSON
    3) Грузим старую модель `model_name` из `models/`
    4) Тренируем заново / дообучаем
    5) Сохраняем как новую версию
    """
    logging.info(f"Переобучение модели '{model_name}' на новом датасете: {dvc_key}")

    datasets_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    try:
        # 1) Скачиваем .dvc
        local_dvc_path = os.path.join(datasets_dir, os.path.basename(dvc_key))
        download_from_s3(dvc_key, local_dvc_path)
        logging.info(f"Downloaded .dvc file from Minio: {dvc_key} -> {local_dvc_path}")

        # 2) dvc pull
        dvc_pull()
        dataset_path = local_dvc_path.replace(".dvc", "")
        if not os.path.exists(dataset_path):
            err = f"После dvc pull не найден {dataset_path}"
            raise HTTPException(status_code=500, detail=err)

        with open(dataset_path, "r") as f:
            data_content = json.load(f)

        from data_validation import Dataset
        dataset = Dataset(**data_content)
        df = dataset.to_dataframe()

        # 3) Грузим старую модель
        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            err = f"Старая модель '{model_name}' не найдена в папке models/"
            raise HTTPException(status_code=404, detail=err)

        model_type = model_name.split("_")[0]  # например, rf_*, catboost_*, etc.
        handler_cls = MODEL_REGISTRY.get(model_type)
        if handler_cls is None:
            err = f"Неизвестный тип модели по имени {model_name}"
            raise HTTPException(status_code=400, detail=err)

        # Загружаем модель (метод load_model должен быть реализован в классе)
        model = handler_cls.load_model(model_name)
        logging.info(f"Модель '{model_name}' загружена для переобучения.")

        # 4) Подготавливаем гиперпараметры для переобучения
        if not hasattr(model, "trained_params") or model.trained_params is None:
            logging.warning("Модель не содержит trained_params, используем дефолт.")
            model.trained_params = {}  # Или можно задать дефолтные параметры

        # 5) Переобучение с логированием в MLflow
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("MyMLProject")

        with mlflow.start_run(run_name=f"Retrain_{model_name}"):
            # Логируем гиперпараметры (используем старые параметры модели)
            for param_name, param_value in model.trained_params.items():
                mlflow.log_param(param_name, param_value)

            # Переобучаем модель
            result = model.train(df, model.trained_params)

            # Логируем метрику, если она возвращается (например, accuracy)
            if "accuracy" in result:
                mlflow.log_metric("accuracy", result["accuracy"])

            # Сохраняем переобученную модель как новую версию
            new_model_name = model.save_model(model_prefix=model_type)
            result["model_name"] = new_model_name
            logging.info(f"Модель переобучена и сохранена под именем: {new_model_name}")

            # Логируем файл модели как артефакт в MLflow
            new_model_path = os.path.join("models", new_model_name)
            if os.path.exists(new_model_path):
                mlflow.log_artifact(new_model_path, artifact_path="custom_model")
                logging.info(f"Файл {new_model_path} залогирован в MLflow как артефакт.")
            else:
                logging.warning(f"Файл модели {new_model_path} не найден, не удалось логировать артефакт.")

        # Конец блока mlflow
        return result

    except Exception as e:
        err = f"Ошибка при переобучении модели: {str(e)}"
        logging.error(err)
        raise HTTPException(status_code=500, detail=err)
    finally:
        clean_datasets_folder()


# Остальные эндпоинты (predict, delete_model, health) оставляем такими как были
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
