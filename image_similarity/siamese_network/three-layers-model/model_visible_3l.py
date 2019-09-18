import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.resnet(x)
        return x


dummy_input = torch.rand(13, 3, 100, 100) #假设输入13张3*100*100的图片
model = Resnet()
with SummaryWriter(comment='Resnet') as w:
    w.add_graph(model, (dummy_input, ))