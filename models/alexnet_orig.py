import torch.nn as nn
import copy

class AlexNetClass(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.activations = []
        self.gradients = []
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 5 * 5), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
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
            x = x.view(x.size(0), 256 * 5 * 5)
            x = self.classifier(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients.append(grad)
    
    def get_activations_gradient(self):
        return self.gradients


def AlexNetOrig(num_classes):
    model = AlexNetClass(num_classes)
    return model