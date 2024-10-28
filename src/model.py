import torch
import torch.nn as nn
from torchvision import models

class CustomModel(nn.Module):
    def __init__(self, architecture, num_classes):
        super(CustomModel, self).__init__()
        self.features = self._make_layers(architecture['conv_layers'])
        self.classifier = self._make_classifier(architecture['fc_layers'], num_classes)

    def _make_layers(self, conv_layers):
        layers = []
        in_channels = 3  # RGB images
        for layer in conv_layers:
            layers.append(nn.Conv2d(in_channels, layer['out_channels'], kernel_size=layer['kernel_size'], padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = layer['out_channels']
        return nn.Sequential(*layers)

    def _make_classifier(self, fc_layers, num_classes):
        layers = []
        in_features = 64 * 16 * 16  # Output size from last conv layer (adjust if image size changes)
        for layer in fc_layers:
            layers.append(nn.Linear(in_features, layer['out_features']))
            layers.append(nn.ReLU())
            in_features = layer['out_features']
        layers.append(nn.Linear(in_features, num_classes))  # Final layer for classification
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x
