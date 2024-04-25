import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import easydict
import math




# argument parser
args = easydict.EasyDict({
"batch_size": 100,
"epochs": 10,
"lr": 0.001,
"enable_cuda": True,
"kernel_sz": 5
})

# class CCPP_CNN(nn.Module):
#     def __init__(self, args, width, height, channels):
#         super(CCPP_CNN, self).__init__()
#         self.features = nn.Sequential()
#         self.features.add_module("conv1", nn.Conv2d(1, 16, kernel_size=args.kernel_sz, stride=1, padding=math.floor(args.kernel_sz/2)))
#         self.features.add_module("bn1", nn.BatchNorm2d(16))
#         self.features.add_module("tanh1", nn.Tanh())
#         self.features.add_module("conv1", nn.Conv2d(16, 32, kernel_size=args.kernel_sz, stride=1, padding=math.floor(args.kernel_sz/2)))
#         self.features.add_module("bn1", nn.BatchNorm2d(32))
#         self.features.add_module("tanh1", nn.Tanh())
#         self.features.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
#         self.features.add_module("conv2", nn.Conv2d(32, 32, kernel_size=args.kernel_sz, stride=1, padding=math.floor(args.kernel_sz/2)))
#         self.features.add_module("bn2", nn.BatchNorm2d(32))
#         self.features.add_module("relu2", nn.Tanh())
#         self.features.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
#         self.linmean1 = nn.Linear(width/4 * height/4 * 32, width/4 * height/4 * 32)
#         self.linmean2 = nn.Linear(width/4 * height/4 * 32, width/4 * height/4 * 32)
#         self.linmeanout = nn.Linear(width/4 * height/4 * 32, 1)
#         self.linvariance1 = nn.Linear(width/4 * height/4 * 32, width/4 * height/4 * 32)
#         self.linvariance2 = nn.Linear(width/4 * height/4 * 32, width/4 * height/4 * 32)
#         self.linvarianceout = nn.Linear(width/4 * height/4 * 32, 1)
#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         mean = self.linmean1(out)
#         mean = self.linmean2(mean)
#         mean = self.linmeanout(mean)
#         variance = self.linvariance1(out)
#         variance = self.linvariance2(variance)
#         variance = self.linvarianceout(variance)
#         variance = torch.exp(variance)
#         return out
class CCPP_CNN(nn.Module):
    def __init__(self, args, width, height, channels):
        super(CCPP_CNN, self).__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.features = nn.Sequential()
        self.features.add_module("lin1", nn.Linear(2048, 2048))
        self.features.add_module("tanh1", nn.Tanh())
        self.features.add_module("lin2", nn.Linear(2048, 1024))
        self.features.add_module("tanh2", nn.Tanh())
        self.features.add_module("lin3", nn.Linear(1024, 3))
        self.model.fc = self.features

    def forward(self, x):
        out = self.model(x)
        return out
    
    