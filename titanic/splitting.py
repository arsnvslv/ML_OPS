"""
Этот скрипт выполняет разбиение данных на train и test,
а также сохраняет их в формате JSON для последующего использования.
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка данных
data = pd.read_csv("titanic/titanic.csv")
data = data.fillna("NaN")

# Разделение данных на признаки и целевую переменную
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Функция для приведения полей Age и Fare к строке
def convert_age_fare_to_str(records):
    for record in records:
        # Приводим Age к строке, если значение не строковое
        if record.get("Age") is not None and not isinstance(record["Age"], str):
            record["Age"] = str(record["Age"])
        # Приводим Fare к строке, если значение не строковое
        if record.get("Fare") is not None and not isinstance(record["Fare"], str):
            record["Fare"] = str(record["Fare"])
    return records

# Формирование JSON-структур
data_train = {
    "X": convert_age_fare_to_str(X_train.to_dict(orient="records")),
    "y": y_train.set_axis(X_train["PassengerId"].values).to_dict()
}
data_test = {
    "X": convert_age_fare_to_str(X_test.to_dict(orient="records"))
}
data_eval = {
    "X": convert_age_fare_to_str(X_test.to_dict(orient="records")),
    "y": y_test.set_axis(X_test["PassengerId"].values).to_dict()
}

# Сохранение данных в JSON-файлы
with open("titanic/train_data.json", "w") as file:
    json.dump(data_train, file)

with open("titanic/test_data.json", "w") as file:
    json.dump(data_test, file)

with open("titanic/eval_data.json", "w") as file:
    json.dump(data_eval, file)

print('Data was successfully splitted!')

