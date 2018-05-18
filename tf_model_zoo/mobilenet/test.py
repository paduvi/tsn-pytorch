import torch
from torch import nn
from pytorch_load import MobileNet

net = MobileNet()

# dynamic change number classes
in_features = net.fc.in_features
net.fc = nn.Linear(in_features, 101)

batch_size = 32
x = torch.randn(batch_size, 3, 340, 256)

y = net(x)
print(y)
