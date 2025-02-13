from pydantic import BaseModel, model_validator
from typing import Optional, List, Dict, Union
import pandas as pd


# Определение структуры данных
class Passenger(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: Union[float, str]
    SibSp: int
    Parch: int
    Ticket: str
    Fare: Union[float, str]
    Cabin: str
    Embarked: str


class Dataset(BaseModel):
    X: List[Passenger]
    y: Optional[Dict[int, int]] = None

    @model_validator(mode="after")  # валидатор выполняется после проверки всех полей
    def check_y_length(self):
        """
        Проверка длины y
        """
        if self.y is not None and len(self.X)!=len(self.y.keys()):
            raise ValueError("Длина y должна быть равна длине X")
        return self  # Возвращаем self, так как Pydantic ожидает изменённый объект

    @model_validator(mode="after")
    def check_keys(self):
        """
        Проверка совдение ключей X и y
        """
        if self.y is not None and set(self.y.keys()) != {p.PassengerId for p in self.X}:
            raise ValueError("Ключи y должны соответствовать PassengerId из X")
        return self

    def to_dataframe(self) -> pd.DataFrame:
        # Преобразуем X в DataFrame
        df_X = pd.DataFrame([p.model_dump() for p in self.X])

        # Если y есть, добавляем его
        if self.y is not None:
            df_X["Survived"] = df_X["PassengerId"].map(self.y)

        return df_X
