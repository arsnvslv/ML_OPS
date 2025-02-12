import json
from validation.data_validation import Dataset

with open('titanic/train_data.json', 'r') as file:
    data = json.load(file)

dataset = Dataset(**data)

