import torch.nn as nn


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1) # 256 * 13 * 13)
    
class AlexNetSplitClass(nn.Module):

    def __init__(self, split_index, num_classes=(1000, 1000)):
        super(AlexNetSplitClass, self).__init__()
        
        self.common, self.task1, self.task2 = make_layers(split_index, num_classes)

    # task is 1 or 2
    def forward(self, x, task=None, activations=False, gradients=False):
        # todo: extract features and gradients
        if task not in [1, 2]:
            raise RuntimeError("Illegal argument: task must be [1, 2]")
        
        x = self.common(x)
        
        if task == 1:
            return self.task1(x)
        else:
            return self.task2(x)

def make_layers(split_index, num_classes):
    num_task1, num_task2 = num_classes
    cfg = [
        ('C', [3, 64, 11, 4, 2]),
        ('B', [64]),
        ('R', []),
        ('C', [64, 192, 5, 1, 2]),
        ('B', [192]),
        ('R', []),
        ('M', [3, 2]),
        ('C', [192, 384, 3, 1, 1]),
        ('B', [384]),
        ('R', []),
        ('C', [384, 256, 3, 1, 1]),
        ('B', [256]),
        ('R', []),
        ('C', [256, 256, 3, 1, 1]),
        ('B', [256]),
        ('R', []),
        ('M', [3, 2]),
        ('V', []),
        ('D', []),
        ('L', [256 * 13 * 13, 4096]),
        ('R', []),
        ('D', []),
        ('L', [4096, 4096]),
        ('R', [])
        ]
    task1_classifier =  [('L', [4096, num_task1])]
    task2_classifier =  [('L', [4096, num_task2])]
    
    common = create_layers(cfg[:split_index])
    task1 = create_layers(cfg[split_index:] + task1_classifier)
    task2 = create_layers(cfg[split_index:] + task2_classifier)
    
    return (common, task1, task2)

def create_layers(cfg):
    layers = []
    for module_type, config in cfg:
        if module_type == 'C':
            i, o, k, s, p = config
            layers.append(nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, stride=s, padding=p))
        elif module_type == 'B':
            batch_size = config[0]
            layers.append(nn.BatchNorm2d(batch_size))
        elif module_type == 'R':
            layers.append(nn.ReLU(inplace=True))
        elif module_type == 'M':
            k, s = config
            layers.append(nn.MaxPool2d(kernel_size=k, stride=s))
        elif module_type == 'D':
            layers.append(nn.Dropout())
        elif module_type == 'L':
            i, o = config
            layers.append(nn.Linear(i, o))
        elif module_type == 'V':
            layers.append(View())
            
    return nn.Sequential(*layers)

def AlexNetSplit(split_index, num_classes):
    model = AlexNetSplitClass(split_index, num_classes)
    return model