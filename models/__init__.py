from models.resnet import resnet18

model_factories = {
    'resnet18': resnet18
}

def get_available_models():
    return model_factories.keys()


def get_model(name, *args, **kwargs):
    return model_factories[name](*args, **kwargs)