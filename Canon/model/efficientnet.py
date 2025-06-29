import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights, EfficientNet_B2_Weights, efficientnet_b2


def get_efficientnet_b1(num_classes=4):
    weights = EfficientNet_B1_Weights.DEFAULT
    model = efficientnet_b1(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def get_efficientnet_b2(num_classes=4):
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
