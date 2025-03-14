"""
gRPC клиент

Подключается к серверу, отправляет запросы на обучение, предсказание,
переобучение, получение списка моделей, проверку состояния и удаление модели.
"""

import grpc
import json
import logging
from google.protobuf import empty_pb2, json_format

import service_pb2
import service_pb2_grpc

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
    logging.FileHandler("grpc_client.log", encoding="utf-8"),
    ]
)

def load_json(file_path):
    """
    Загружает JSON-данные из файла.
    """
    with open(file_path, "r") as f:
        return json.load(f)

def main():
    """
    Основная функция клиента. Выполняет серию запросов к gRPC серверу.
    """
    logging.info("Устанавливаем соединение с gRPC-сервером на localhost:50051")
    channel = grpc.insecure_channel("localhost:50051")
    client = service_pb2_grpc.ModelServiceStub(channel)

    # 1. TrainModel
    train_data = load_json("titanic/train_data.json")
    train_request_dict = {
        "model_type": "catboost",
        "hyperparameters": {"iterations": 1000, "depth": 6},
        "dataset": train_data
    }
    train_request = json_format.ParseDict(train_request_dict, service_pb2.TrainRequest())
    logging.info("Отправляем запрос TrainModel")
    train_response = client.TrainModel(train_request)
    logging.info("TrainModel response: %s", train_response)
    print("TrainModel response:")
    print(train_response)

    # 2. Predict
    test_data = load_json("titanic/test_data.json")
    predict_request_dict = {
        "model_name": train_response.model_name,
        "dataset": test_data
    }
    predict_request = json_format.ParseDict(predict_request_dict, service_pb2.PredictRequest())
    logging.info("Отправляем запрос Predict для модели: %s", train_response.model_name)
    predict_response = client.Predict(predict_request)
    logging.info("Predict response: %s", predict_response)
    print("\nPredict response:")
    print(predict_response)

    # 3. Retrain
    eval_data = load_json("titanic/eval_data.json")
    retrain_request_dict = {
        "model_name": train_response.model_name,
        "dataset": eval_data
    }
    retrain_request = json_format.ParseDict(retrain_request_dict, service_pb2.RetrainRequest())
    logging.info("Отправляем запрос Retrain для модели: %s", train_response.model_name)
    retrain_response = client.Retrain(retrain_request)
    logging.info("Retrain response: %s", retrain_response)
    print("\nRetrain response:")
    print(retrain_response)

    # 4. ListTrainedModels
    logging.info("Отправляем запрос ListTrainedModels")
    list_models_response = client.ListTrainedModels(empty_pb2.Empty())
    logging.info("ListTrainedModels response: %s", list_models_response)
    print("\nListTrainedModels response:")
    print(list_models_response)

    # 5. ListAvailableModels
    logging.info("Отправляем запрос ListAvailableModels")
    list_available_response = client.ListAvailableModels(empty_pb2.Empty())
    logging.info("ListAvailableModels response: %s", list_available_response)
    print("\nListAvailableModels response:")
    print(list_available_response)

    # 6. HealthCheck
    logging.info("Отправляем запрос HealthCheck")
    health_response = client.HealthCheck(empty_pb2.Empty())
    logging.info("HealthCheck response: %s", health_response)
    print("\nHealthCheck response:")
    print(health_response)

    # 7. DeleteModel
    delete_request = service_pb2.DeleteModelRequest(model_name=train_response.model_name)
    logging.info("Отправляем запрос DeleteModel для модели: %s", train_response.model_name)
    delete_response = client.DeleteModel(delete_request)
    logging.info("DeleteModel response: %s", delete_response)
    print("\nDeleteModel response:")
    print(delete_response)

if __name__ == "__main__":
    main()
