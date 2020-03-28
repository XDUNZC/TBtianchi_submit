import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torchvision
import numpy as np


x = torch.rand([32, 2048, 4, 4])
# x = x.view(x.size(0), -1)
print(x.shape)
x.avgpool = nn.AvgPool2d(3, stride=1)
x = x.avgpool(x)
print(x.shape)
