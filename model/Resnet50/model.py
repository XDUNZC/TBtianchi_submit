import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torchvision
import numpy as np


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Siamese_ResNet(nn.Module):
    def __init__(self,blocks, num_classes=1, expansion = 4):
        super(Siamese_ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc1 = nn.Linear(8192, 2048) #105*105: 8192
        self.fc2 = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward_one(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        # import pdb; pdb.set_trace()
        x = self.avgpool(x)
        # print(x.shape)
        # import pdb; pdb.set_trace()
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.fc2(dis)
        #  return self.sigmoid(out)
        return out



if __name__=='__main__':
    # net = Siamese_ResNet([3, 4, 6, 3])
    # #model = torchvision.models.resnet50()
    # model = ResNet50()
    # print(model)

    # input = torch.randn(1, 3, 224, 224)
    # out = model(input)
    # print(out.shape)

    net = Siamese_ResNet([3, 4, 6, 3])
    print(net)

# class ResidualBlock(nn.Module):
#     def __init__(self, inchannel, outchannel, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(outchannel)
#             )

#     def forward(self, x):
#         out = self.left(x)
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class Siamese(nn.Module):
#     def __init__(self, ResidualBlock, num_classes=1):
#         super(Siamese, self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
#         self.fc1 = nn.Linear(4608, 4096)
#         self.fc2 = nn.Linear(4096, num_classes)

#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         return nn.Sequential(*layers)

#     def forward_one(self, x):
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = F.avg_pool2d(x, 4)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         # x = self.fc(x)
#         return x
#     def forward(self, x1, x2):
#         out1 = self.forward_one(x1)
#         out2 = self.forward_one(x2)
#         dis = torch.abs(out1 - out2)
#         out = self.fc2(dis)
#         #  return self.sigmoid(out)
#         return out


# # class Siamese(nn.Module):

# #     def __init__(self):
# #         super(Siamese, self).__init__()
# #         self.conv = nn.Sequential(
# #             nn.Conv2d(3, 64, 10),  # 64@96*96
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(2),  # 64@48*48
# #             nn.Conv2d(64, 128, 7),
# #             nn.ReLU(),    # 128@42*42
# #             nn.MaxPool2d(2),   # 128@21*21
# #             nn.Conv2d(128, 128, 4),
# #             nn.ReLU(), # 128@18*18
# #             nn.MaxPool2d(2), # 128@9*9
# #             nn.Conv2d(128, 256, 4),
# #             nn.ReLU(),   # 256@6*6
# #         )
# #         self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
# #         self.out = nn.Linear(4096, 1)

# #     def forward_one(self, x):
# #         x = self.conv(x)
# #         x = x.view(x.size()[0], -1)
# #         x = self.liner(x)
# #         return x

# #     def forward(self, x1, x2):
# #         out1 = self.forward_one(x1)
# #         out2 = self.forward_one(x2)
# #         dis = torch.abs(out1 - out2)
# #         out = self.out(dis)
# #         #  return self.sigmoid(out)
# #         return out


# # for test
# if __name__ == '__main__':
#     # resnet18 = models.resnet18(pretrained=True)
#     # pretrained_dict =resnet18.state_dict() 
#     # model_dict = model.state_dict() 

#     '''
#     # net = Siamese()
#     # print(net)
#     # print(list(net.parameters()))
#     '''
#     #1、 using Resnet18 to fine tuning models
#     # net = torchvision.models.resnet18(pretrained=True)#加载已经训练好的模型
#     # num_ftrs = net.fc.in_features
#     # net.fc = nn.Linear(num_ftrs, 1)#将全连接层做出改变类别改为一类
#     # print(net)
#     # print(list(net.parameters()))

#     #2、 fix convolution to train fully connected layer
#     # net = torchvision.models.resnet18(pretrained=True)

#     # for param in net.parameters():
#     #     param.requires_grad = False

#     # num_ftrs = net.fc.in_features
#     # net.fc = nn.Linear(num_ftrs, 1)
#     # print(net)
#     # print(list(net.parameters()))

#     net = Siamese(ResidualBlock)
#     print(net)