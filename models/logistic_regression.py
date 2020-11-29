from models.mlp import get_mlp

def get_logistic(input_dim, num_classes=10):
    return get_mlp(input_dim, num_classes, hidden_units=[])