from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, List, Dict, Union


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

    @field_validator("y")
    @classmethod
    def check_y_length(cls, y, values):
        if y is not None and "X" in values and len(y) != len(values["X"]):
            raise ValueError("Длина y должна быть равна длине X")
        return y

    @model_validator(mode="after")  # валидатор выполняется после проверки всех полей
    def check_keys(self):
        if self.y is not None and set(self.y.keys()) != {p.PassengerId for p in self.X}:
            raise ValueError("Ключи y должны соответствовать PassengerId из X")
        return self  # Возвращаем self, так как Pydantic ожидает изменённый объект
