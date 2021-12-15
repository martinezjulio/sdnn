'''
Created on Mar 11, 2019
@author: iapalm
'''
import torch.nn as nn
# from _ast import Num


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)

class VGGSplitClass(nn.Module):

    def __init__(self, split_index, num_classes=(1000, 1000), init_weights=True):
        super(VGGSplitClass, self).__init__()
        
        self.common, self.task1, self.task2 = make_layers(split_index, num_classes)
        
        if init_weights:
            self._initialize_weights()

    # task is 1 or 2
    def forward(self, x, task, features_layer=None, classifier_layer=None):
        if task not in [1, 2]:
            raise RuntimeError("Illegal argument: task must be [1, 2]")
        
        x = self.common(x)
        
        if task == 1:
            return self.task1(x)
        else:
            return self.task2(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(split_index, num_classes, batch_norm=False):
    num_task1, num_task2 = num_classes
    cfg = [64,  64,  'M', 
           128, 128, 'M', 
           256, 256, 256, 'M', 
           512, 512, 512, 'M', 
           512, 512, 512, 'M', 
           'V', 
           (512 * 7 * 7, 4096), 'R', 'D', 
           (4096, 4096), 'R', 'D']
    
    task1_classifier =  [(4096, num_task1)]
    task2_classifier =  [(4096, num_task2)]
    
    in_channels, common = create_layer(cfg[:split_index], 3, batch_norm)
    _, task1 = create_layer(cfg[split_index:] + task1_classifier, in_channels, batch_norm)
    _, task2 = create_layer(cfg[split_index:] + task2_classifier, in_channels, batch_norm)
    
    return (common, task1, task2)
    
def create_layer(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'V':
            layers += [View()]
        elif v == 'R':
            layers += [nn.ReLU(inplace=True)]
        elif v == 'D':
            layers += [nn.Dropout()]
        elif isinstance(v, tuple):
            layers += [nn.Linear(v[0], v[1])]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    return (in_channels, nn.Sequential(*layers))


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def VGG16Split(split_index, num_classes):
    """VGG 16-layer model (configuration "D")"""
    model = VGGSplitClass(split_index, num_classes)
    return model