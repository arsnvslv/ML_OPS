from base_model import BaseMlModel

import os
from datetime import datetime
from typing import Type, Dict, Any, Optional,List

import joblib
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier


#########################################
# 1. Регистрация обработчиков моделей
#########################################


# Глобальный реестр для обработчиков моделей
MODEL_REGISTRY: Dict[str, Type["BaseMlModel"]] = {}


def register_model_handler(model_type: str):
    """
    Декоратор для регистрации обработчика модели.
    Проверяет, что класс наследуется от BaseModelHandler и содержит атрибут params_schema (наследник BaseModel).
    """
    def decorator(cls: Type["BaseMlModel"]):
        if not issubclass(cls, BaseMlModel):
            raise ValueError(f"Класс {cls.__name__} должен наследоваться от BaseModelHandler")
        if not hasattr(cls, "params_schema") or not issubclass(cls.params_schema, BaseModel):
            raise ValueError(
                f"Класс {cls.__name__} должен определить атрибут params_schema, наследуемый от BaseModel"
            )
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator


class RandomForestParams(BaseModel):
    n_estimators: Optional[int] = Field(100, gt=0, description="Количество деревьев")
    max_depth: Optional[int] = Field(10, gt=0, description="Максимальная глубина дерева")
    random_state: int = Field(42, description="Сид генератора случайных чисел")


@register_model_handler("rf")
class RandomForestHandler(BaseMlModel):
    """Модель Random Forest"""
    params_schema = RandomForestParams

    def _train(self, X, y, params: dict) -> Dict[str, Any]:
        clf = RandomForestClassifier(**params)
        clf.fit(X, y)
        self.model = clf
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        # Сохраняем параметры для возможного переобучения
        self.trained_params = params
        return {"accuracy": acc, "model_prefix": "rf"}

    def _predict(self, X) -> List[Any]:
        return self.model.predict(X).tolist()

    def save_model(self, model_prefix: str) -> str:
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_prefix}_{timestamp}.pkl"
        model_path = os.path.join("models", model_name)
        # Создаём пайплайн с препроцессором и моделью
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        # Сохраняем пайплайн и обучающие параметры в один объект
        model_info = {
            "pipeline": pipeline,
            "trained_params": self.trained_params if hasattr(self, "trained_params") else None
        }
        joblib.dump(model_info, model_path)
        return model_name

    @classmethod
    def load_model(cls, model_name: str) -> "RandomForestHandler":
        model_path = os.path.join("models", model_name)
        model_info = joblib.load(model_path)
        # Создаём новый экземпляр обработчика
        instance = cls()
        pipeline: Pipeline = model_info.get("pipeline")
        # Восстанавливаем preprocessor и модель из пайплайна
        instance.preprocessor = pipeline.named_steps["preprocessor"]
        instance.model = pipeline.named_steps["classifier"]
        instance.trained_params = model_info.get("trained_params")
        return instance


# Pydantic-схема гиперпараметров для CatBoost
class CatBoostParams(BaseModel):
    iterations: Optional[int] = Field(None, gt=0, description="Количество итераций обучения")
    depth: Optional[int] = Field(None, gt=0, description="Глубина дерева")
    learning_rate: Optional[float] = Field(None, gt=0, description="Скорость обучения")
    random_seed: int = Field(42, description="Сид генератора случайных чисел")
    verbose: bool = Field(False, description="Выводить ли промежуточные результаты")


# Регистрация обработчика (используем тот же декоратор register_model_handler)
@register_model_handler("catboost")
class CatBoost(BaseMlModel):
    """Модель CatBoost"""
    params_schema = CatBoostParams

    def _train(self, X, y, params: dict) -> Dict[str, Any]:
        model = CatBoostClassifier(**params)
        model.fit(X, y, verbose=params.get("verbose", False))
        self.model = model
        preds = model.predict(X)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y, preds)
        # Сохраняем параметры для переобучения
        self.trained_params = params
        return {"accuracy": acc, "model_prefix": "catboost"}

    def _predict(self, X) -> List[Any]:
        return self.model.predict(X).tolist()

    def save_model(self, model_prefix: str) -> str:
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_prefix}_{timestamp}.pkl"
        model_path = os.path.join("models", model_name)
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        model_info = {
            "pipeline": pipeline,
            "trained_params": self.trained_params if hasattr(self, "trained_params") else None
        }
        joblib.dump(model_info, model_path)
        return model_name

    @classmethod
    def load_model(cls, model_name: str) -> "CatBoost":
        model_path = os.path.join("models", model_name)
        model_info = joblib.load(model_path)
        instance = cls()
        pipeline: Pipeline = model_info.get("pipeline")
        instance.preprocessor = pipeline.named_steps["preprocessor"]
        instance.model = pipeline.named_steps["classifier"]
        instance.trained_params = model_info.get("trained_params")
        return instance

