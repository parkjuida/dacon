import json


def load_experiment(index):
    with open(f"experiments/lightgbm/{index}.json") as f:
        return json.load(f)
