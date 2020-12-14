from models.resnet import resnet18
from models.mlp import get_mlp
from models.logistic_regression import get_logistic


model_factories = {
    'resnet18': resnet18,
    'mlp': get_mlp,
    'logistic': get_logistic,
}

def get_available_models():
    return model_factories.keys()


def get_model(name, *args, **kwargs):
    return model_factories[name](*args, **kwargs)