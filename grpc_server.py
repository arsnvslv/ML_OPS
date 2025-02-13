"""
gRPC сервер

Реализует методы для обучения, предсказания, переобучения, удаления моделей,
а также методы для получения списка доступных/обученных моделей и проверки
здоровья системы.
"""

import grpc
from concurrent import futures
import time
import os
import json
import numpy as np
import logging

from google.protobuf import empty_pb2, struct_pb2, json_format

import service_pb2
import service_pb2_grpc

# Импорт рефлексии для gRPC
from grpc_reflection.v1alpha import reflection

from health import check_system_health
from data_validation import Dataset as PydanticDataset, Passenger as PydanticPassenger
from models import MODEL_REGISTRY

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        # Если требуется, можно добавить FileHandler для записи в файл
    ]
)

def proto_passenger_to_dict(proto_passenger):
    """
    Преобразует protobuf-сообщение Passenger в словарь.
    """
    return {
        "PassengerId": proto_passenger.passenger_id,
        "Pclass": proto_passenger.pclass,
        "Name": proto_passenger.name,
        "Sex": proto_passenger.sex,
        "Age": proto_passenger.age,      # Уже строка
        "SibSp": proto_passenger.sibsp,
        "Parch": proto_passenger.parch,
        "Ticket": proto_passenger.ticket,
        "Fare": proto_passenger.fare,    # Уже строка
        "Cabin": proto_passenger.cabin,
        "Embarked": proto_passenger.embarked,
    }

def proto_dataset_to_dict(proto_dataset):
    """
    Преобразует protobuf Dataset в словарь с данными.
    """
    passengers = [proto_passenger_to_dict(p) for p in proto_dataset.X]
    labels = dict(proto_dataset.y) if proto_dataset.y else None
    return {"X": passengers, "y": labels}

class ModelServiceServicer(service_pb2_grpc.ModelServiceServicer):
    """
    Сервис для работы с моделями машинного обучения.
    Реализует методы обучения, предсказания, переобучения и управления моделями.
    """

    def TrainModel(self, request, context):
        """
        Обучает модель заданного типа с использованием предоставленного датасета и гиперпараметров.
        """
        logging.info("Получен запрос TrainModel для модели: %s", request.model_type)
        model_type = request.model_type
        if model_type not in MODEL_REGISTRY:
            error_msg = f"Тип модели '{model_type}' не поддерживается."
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return service_pb2.TrainResponse()

        handler_cls = MODEL_REGISTRY.get(model_type)

        # Валидация гиперпараметров через Pydantic
        try:
            hyperparams_dict = json_format.MessageToDict(request.hyperparameters)
            validated_params = handler_cls.params_schema(**hyperparams_dict)
        except Exception as e:
            error_msg = f"Ошибка в параметрах модели: {str(e)}"
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return service_pb2.TrainResponse()

        # Обработка данных
        try:
            dataset_dict = proto_dataset_to_dict(request.dataset)
            dataset_obj = PydanticDataset(**dataset_dict)
            df = dataset_obj.to_dataframe()
            if dataset_obj.y is None:
                error_msg = "Отсутствует целевая переменная 'y' в данных."
                logging.error(error_msg)
                context.set_details(error_msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return service_pb2.TrainResponse()
        except Exception as e:
            error_msg = f"Ошибка в данных: {str(e)}"
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return service_pb2.TrainResponse()

        # Обучение модели
        try:
            model_instance = handler_cls()
            result = model_instance.train(df, validated_params.dict())
            model_prefix = result.get("model_prefix", "model")
            model_name = model_instance.save_model(model_prefix)
            accuracy = result.get("accuracy", 0.0)
            logging.info("Модель успешно обучена. model_name: %s, accuracy: %s", model_name, accuracy)
        except Exception as e:
            error_msg = f"Ошибка при обучении модели: {str(e)}"
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.TrainResponse()

        return service_pb2.TrainResponse(
            accuracy=accuracy,
            model_prefix=model_prefix,
            model_name=model_name
        )

    def Predict(self, request, context):
        """
        Делает предсказание на основе предоставленного датасета и указанной модели.
        """
        logging.info("Получен запрос Predict для модели: %s", request.model_name)
        model_name = request.model_name
        try:
            dataset_dict = proto_dataset_to_dict(request.dataset)
            dataset_dict.pop("y", None)
            dataset_obj = PydanticDataset(**dataset_dict)
        except Exception as e:
            error_msg = f"Ошибка в данных: {str(e)}"
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return service_pb2.PredictResponse()

        try:
            model_type = model_name.split("_")[0]
        except Exception as e:
            error_msg = "Некорректный формат имени модели."
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return service_pb2.PredictResponse()

        handler_cls = MODEL_REGISTRY.get(model_type)
        if handler_cls is None:
            error_msg = f"Тип модели '{model_type}' не поддерживается."
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return service_pb2.PredictResponse()

        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            error_msg = f"Модель '{model_name}' не найдена."
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return service_pb2.PredictResponse()

        try:
            model_instance = handler_cls.load_model(model_name)
            df = dataset_obj.to_dataframe()
            predictions = model_instance.predict(df)
            pred_dict = {}
            passenger_ids = [p["PassengerId"] for p in dataset_dict["X"]]
            for pid, pred in zip(passenger_ids, predictions):
                if isinstance(pred, (np.integer,)):
                    pred_dict[int(pid)] = int(pred)
                elif isinstance(pred, (np.floating,)):
                    pred_dict[int(pid)] = float(pred)
                else:
                    pred_dict[int(pid)] = pred
            logging.info("Предсказание успешно выполнено для модели: %s", model_name)
        except Exception as e:
            error_msg = f"Ошибка при предсказании: {str(e)}"
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.PredictResponse()

        return service_pb2.PredictResponse(predictions=pred_dict)

    def Retrain(self, request, context):
        """
        Переобучает модель на новом датасете.
        """
        logging.info("Получен запрос Retrain для модели: %s", request.model_name)
        model_name = request.model_name
        try:
            dataset_dict = proto_dataset_to_dict(request.dataset)
            dataset_obj = PydanticDataset(**dataset_dict)
            df = dataset_obj.to_dataframe()
            if dataset_obj.y is None:
                error_msg = "Отсутствует целевая переменная 'y' в данных."
                logging.error(error_msg)
                context.set_details(error_msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return service_pb2.RetrainResponse()
        except Exception as e:
            error_msg = f"Ошибка в данных: {str(e)}"
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return service_pb2.RetrainResponse()

        try:
            model_type = model_name.split("_")[0]
        except Exception as e:
            error_msg = "Некорректный формат имени модели."
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return service_pb2.RetrainResponse()

        handler_cls = MODEL_REGISTRY.get(model_type)
        if handler_cls is None:
            error_msg = f"Тип модели '{model_type}' не поддерживается."
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return service_pb2.RetrainResponse()

        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            error_msg = f"Модель '{model_name}' не найдена."
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return service_pb2.RetrainResponse()

        try:
            model_instance = handler_cls.load_model(model_name)
        except Exception as e:
            error_msg = f"Ошибка при загрузке модели: {str(e)}"
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.RetrainResponse()

        if not hasattr(model_instance, "trained_params") or model_instance.trained_params is None:
            error_msg = "Параметры модели не найдены для переобучения."
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.RetrainResponse()

        try:
            result = model_instance.train(df, model_instance.trained_params)
            new_model_name = model_instance.save_model(model_prefix=model_type)
            accuracy = result.get("accuracy", 0.0)
            logging.info("Модель переобучена. new_model_name: %s, accuracy: %s", new_model_name, accuracy)
        except Exception as e:
            error_msg = f"Ошибка при переобучении модели: {str(e)}"
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.RetrainResponse()

        return service_pb2.RetrainResponse(
            accuracy=accuracy,
            model_name=new_model_name
        )

    def ListTrainedModels(self, request, context):
        """
        Возвращает список файлов обученных моделей.
        """
        logging.info("Получен запрос ListTrainedModels")
        models_dir = 'models'
        if not os.path.exists(models_dir):
            error_msg = "Папка моделей не найдена."
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return service_pb2.ListModelsResponse()
        try:
            models_list = [
                f for f in os.listdir(models_dir)
                if os.path.isfile(os.path.join(models_dir, f))
            ]
            logging.info("Найдено моделей: %d", len(models_list))
        except Exception as e:
            error_msg = "Ошибка получения списка моделей."
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.ListModelsResponse()
        return service_pb2.ListModelsResponse(models=models_list)

    def DeleteModel(self, request, context):
        """
        Удаляет модель с указанным именем.
        """
        logging.info("Получен запрос DeleteModel для модели: %s", request.model_name)
        model_path = os.path.join("models", request.model_name)
        if not os.path.exists(model_path):
            error_msg = f"Модель '{request.model_name}' не найдена."
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return service_pb2.DeleteModelResponse()
        try:
            os.remove(model_path)
            logging.info("Модель '%s' успешно удалена.", request.model_name)
        except Exception as e:
            error_msg = f"Ошибка при удалении модели: {str(e)}"
            logging.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.DeleteModelResponse()
        return service_pb2.DeleteModelResponse(detail=f"Модель '{request.model_name}' успешно удалена.")

    def HealthCheck(self, request, context):
        """
        Возвращает информацию о состоянии системы.
        """
        logging.info("Получен запрос HealthCheck")
        health = check_system_health()
        return service_pb2.HealthResponse(
            cpu_usage=health.get("cpu_usage", ""),
            memory_usage=health.get("memory_usage", ""),
            disk_usage=health.get("disk_usage", "")
        )

    def ListAvailableModels(self, request, context):
        """
        Возвращает список доступных типов моделей с описаниями и схемами гиперпараметров.
        """
        logging.info("Получен запрос ListAvailableModels")
        available = []
        for model_type, handler_cls in MODEL_REGISTRY.items():
            available.append(service_pb2.AvailableModel(
                model_type=model_type,
                description=handler_cls.__doc__ or "Нет описания",
                params_schema=json.dumps(handler_cls.params_schema.schema())
            ))
        return service_pb2.ListAvailableModelsResponse(models=available)

def serve():
    """
    Запускает gRPC сервер на порту 50051 с включенной рефлексией.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_ModelServiceServicer_to_server(ModelServiceServicer(), server)

    SERVICE_NAMES = (
        service_pb2.DESCRIPTOR.services_by_name['ModelService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port('[::]:50051')
    logging.info("gRPC сервер запущен на порту 50051")
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logging.info("Остановка сервера по KeyboardInterrupt")
        server.stop(0)

if __name__ == '__main__':
    serve()

