from models import MODEL_REGISTRY
import json
from data_validation import Dataset

with open('titanic/train_data.json', 'r') as file:
    data = json.load(file)

dataset = Dataset(**data)
dataset = dataset.to_dataframe()

params = {'max_depth': 5, 'random_state': 42}

model = MODEL_REGISTRY['catboost']()
result = model.train(data = dataset, params = params)


model_prefix = result.get("model_prefix", "model")
model_name = model.save_model(model_prefix)
result["model_name"] = model_name

with open('titanic/test_data.json', 'r') as file:
    data = json.load(file)

dataset = Dataset(**data)
dataset = dataset.to_dataframe()

model = MODEL_REGISTRY['rf']()
model = model.load_model(model_name=model_name)
print(model.predict(dataset))






# model_prefix = result.get("model_prefix", "model")
# model_name = self.save_model(model_prefix)
# result["model_name"] = model_name