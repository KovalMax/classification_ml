from typing import TypedDict
import toml


class ConfigMap(TypedDict):
    seed_number: int
    samples: int
    number_of_clusters: int
    number_of_features: int
    batch_size: int
    hidden_layer_neurons: int
    number_epochs: int
    activation_hidden: str
    activation_classification: str
    optimizer: str
    loss: str


class ApplicationConfig:
    def __init__(self):
        with open('config.toml', 'r') as file:
            self.__entries: ConfigMap = toml.loads(file.read())

    def seed_number(self) -> int:
        return self.__entries['seed_number']

    def samples(self) -> int:
        return self.__entries['samples']

    def number_of_clusters(self) -> int:
        return self.__entries['number_of_clusters']

    def number_of_features(self) -> int:
        return self.__entries['number_of_features']

    def batch_size(self) -> int:
        return self.__entries['batch_size']

    def number_epochs(self) -> int:
        return self.__entries['number_epochs']

    def activation_hidden(self) -> str:
        return self.__entries['activation_hidden']

    def activation_classification(self) -> str:
        return self.__entries['activation_classification']

    def optimizer(self) -> str:
        return self.__entries['optimizer']

    def loss(self) -> str:
        return self.__entries['loss']

    def hidden_layer_neurons(self) -> int:
        return self.__entries['hidden_layer_neurons']
