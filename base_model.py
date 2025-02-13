from abc import ABC, abstractmethod
from typing import Dict, Any, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class BaseMlModel(ABC):
    """
    Базовый класс для всех обработчиков моделей.

    Выполняет общий препроцессинг данных, а затем вызывает
    абстрактные методы _train() и _predict(), реализуемые в потомках.

    Абстрактные методы:
      - build_pipeline(): возвращает объект для сохранения модели.
         Для sklearn‑моделей – обычно Pipeline, для других – может быть любая структура.
      - save_model(): сохраняет полученный объект.
      - load_model(): загружает сохранённую модель.
    """
    # Списки признаков, обрабатываемые одинаково для всех моделей
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']

    def __init__(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), self.numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), self.categorical_features)
            ]
        )
        self.model = None  # Здесь будет храниться обученная модель

    def train(self, data: pd.DataFrame, params: dict) -> Dict[str, Any]:
        """
        Общий метод обучения модели.
        1. Фитит preprocessor и обрабатывает данные.
        2. Вызывает _train() для обучения модели на обработанных данных.
        """
        X = data.drop(columns=['Survived'])
        y = data['Survived']
        X_processed = self.preprocessor.fit_transform(X)
        result = self._train(X_processed, y, params)
        # Получаем объект для сохранения (например, Pipeline для sklearn‑моделей)
        return result

    @abstractmethod
    def _train(self, X, y, params: dict) -> Dict[str, Any]:
        """
        Обучает модель на обработанных данных.
        Должен установить self.model и вернуть информацию (например, accuracy, префикс для имени модели).
        """
        pass

    def predict(self, data: pd.DataFrame) -> List[Any]:
        """
        Общий метод предсказания.
        Применяет preprocessor (transform) к данным и вызывает _predict().
        """
        X_processed = self.preprocessor.transform(data)
        return self._predict(X_processed)

    @abstractmethod
    def _predict(self, X) -> List[Any]:
        """
        Делает предсказание на обработанных данных.
        """
        pass

    @abstractmethod
    def save_model(self, model_prefix: str) -> str:
        """
        Сохраняет объект модели (например, Pipeline или другую структуру) и возвращает имя файла.
        """
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, model_name: str) -> Any:
        """
        Загружает объект модели по имени файла.
        """
        pass