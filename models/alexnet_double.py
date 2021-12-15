import torch.nn as nn
import copy

class AlexNetDoubleClass(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.activations = []
        self.gradients = []
        self.features = nn.Sequential(
            nn.Conv2d(3, 64*2, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(64*2, 192*2, kernel_size=5, padding=2),
            nn.BatchNorm2d(192*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192*2, 384*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(384*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(384*2, 256*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(256*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256*2, 256*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(256*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 13 * 13*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, features_layer=None, activations=False, gradients=False):
        self.activations = []
        if activations:
            num_features = len(self.features)
            num_classifier = len(self.classifier)
            for i in range(num_features):
                x = self.features[i](x)
                self.activations.append(copy.deepcopy(x))
                if gradients:
                    x.register_hook(self.activations_hook)
            x = x.view(x.size(0), -1)
            for i in range(num_classifier):
                x = self.classifier[i](x)
                self.activations.append(copy.deepcopy(x))
                if gradients:
                    x.register_hook(self.activations_hook)
            
        else:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients.append(grad)
    
    def get_activations_gradient(self):
        return self.gradients


def AlexNetDouble(num_classes):
    model = AlexNetDoubleClass(num_classes)
    return model
