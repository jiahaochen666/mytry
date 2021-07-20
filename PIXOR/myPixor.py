import torch
import torch.nn as nn
import os
from my_load_data import *


class Residual(nn.Module):
    def __init__(self, input, output, stride=1, down_sample=False):
        super(Residual, self).__init__()
        self.input = input
        self.output = output
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=False),
            nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=False),
            nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=False)
        )
        self.down_sample = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=1, stride=2),
            nn.BatchNorm2d(output)
        )
        if not down_sample:
            self.down_sample = None

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.down_sample:
            x += self.down_sample(residual)
        x = self.relu(x)
        return x


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(36, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )

        self.block2 = self.make_residual_layer(32, 96, 3)
        self.block3 = self.make_residual_layer(96, 192, 6)
        self.block4 = self.make_residual_layer(192, 256, 6)
        self.block5 = self.make_residual_layer(256, 384, 3)

        self.layer1 = nn.Conv2d(384, 196, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(256, 128, kernel_size=1,stride=1, padding=0)
        self.layer3 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)

        self.deconv1 = nn.ConvTranspose2d(196, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=(0, 1))

    def make_residual_layer(self, input, output, num_layers):
        layer = []
        for i in range(num_layers):
            if i == 0:
                layer.append(Residual(input, output // 4, 2, True))
            elif i == num_layers - 1:
                layer.append(Residual(output // 4, output))
            else:
                layer.append(Residual(output // 4, output // 4))
        return nn.Sequential(*layer)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        l5 = self.layer1(x5)
        l4 = self.layer2(x4)
        p5 = l4 + self.deconv1(l5)
        l3 = self.layer3(x3)
        p5 = self.deconv2(p5)
        p5 = l3 + p5

        return p5

class PIXOR(nn.Module):
    def __init__(self):
        super(PIXOR, self).__init__()
        self.backbone = Backbone()
        self.header = self.make_header()
        self.classification = nn.Conv2d(96, 1, kernel_size=(3, 3), padding=(1, 1))
        self.regression = nn.Conv2d(96, 6, kernel_size=(3, 3), padding=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def make_header(self):
        layer = []
        for i in range(4):
            layer.append(nn.Conv2d(96, 96, kernel_size=(3, 3), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(96))
            layer.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.backbone(x)
        x = self.header(x)
        c = self.sigmoid(self.classification(x))
        r = self.regression(x)
        x = torch.cat((r, c), dim=1)
        return x