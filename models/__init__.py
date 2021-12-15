import torch

from models.alexnet import AlexNet
from models.alexnet_nobatchnorm import AlexNetNoBatchnorm
from models.alexnet_orig import AlexNetOrig
from models.alexnet_default import AlexNetDefault
from models.alexnet_double import AlexNetDouble
from models.alexnet_split import AlexNetSplit

from models.vgg import VGG16
from models.vgg_split import VGG16Split
from models.vgg_subnet import VGG16SubNetwork

from models.resnet import resnet50 as ResNet50


#from models.vgg_subnet import VGG16LotteryTicket

#from models.vgg_sub import VGG16 as VGG16Sub

#def vgg16(num_classes):
#    model = VGG16(num_classes=num_classes)
#    model = torch.nn.DataParallel(model)
#    return model
#
#def alexnet(num_classes):
#    model = AlexNet(num_classes=num_classes)
#    model = model = torch.nn.DataParallel(model)
#    return model
#
#def vgg16_split(split_index, num_classes):
#    model = VGG16Split(split_index=split_index, num_classes=num_classes)
#    model = torch.nn.DataParallel(model)
#    return model
#
#def alexnet_split(split_index, num_classes):
#    model = AlexNetSplit(split_index=split_index, num_classes=num_classes)
#    #print('no DataParallel')
#    model = torch.nn.DataParallel(model)
#    return model
