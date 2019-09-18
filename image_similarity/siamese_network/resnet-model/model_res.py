import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        model = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(model.children())[:-2])
        self.conv2 = nn.Sequential(
            nn.Conv2d(2048, 100, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc1 = nn.Linear(100, 100)
        ct = 0
        for child in self.resnet.children():
            ct += 1
            if ct < 3:  # 1-6 fixed
                for param in child.parameters():
                    param.requires_grad = False

    def forward_once(self, x):
        x = self.resnet(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # loss_contrastive = torch.mean(
        #     (1 - label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(
        #     self.margin - euclidean_distance, min=0.0), 2))

        # 0 for same and 1 for difference
        loss_contrastive = torch.mean(
            (1 - label) * 10 * torch.pow(euclidean_distance, 2) + (label) * 50/(torch.clamp(euclidean_distance, max=8)+0.00001)
        )

        return loss_contrastive


# class CNN(nn.Module):
#
#     def __init__(self, block, layers):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.conv2 = nn.Conv2d(2048, 5, kernel_size=1, stride=1)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         # 新加层的forward
#         x=self.conv2(x)
#         x = x.view(x.size(0), -1)
#
#         return x
#
#
# # 加载model
# resnet50 = models.resnet50(pretrained=True)
# cnn = CNN(Bottleneck, [3, 4, 6, 3])
# # 读取参数
# pretrained_dict = resnet50.state_dict()
# model_dict = cnn.state_dict()
# # 将pretrained_dict里不属于model_dict的键剔除掉
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # 更新现有的model_dict
# model_dict.update(pretrained_dict)
# # 加载我们真正需要的state_dict
# cnn.load_state_dict(model_dict)
# # print(resnet50)
# print(cnn)