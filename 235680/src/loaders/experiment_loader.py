import json


def load_experiment_lightgbm(index):
    with open(f"experiments/lightgbm/{index}.json") as f:
        return json.load(f)


def load_experiment_cnn(index):
    with open(f"experiments/cnn/{index}.json") as f:
        return json.load(f)