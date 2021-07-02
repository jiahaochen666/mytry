import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(36, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.block2 = self.make_residual_layer(32, 96, 3)
        self.block3 = self.make_residual_layer(96, 196, 6)
        self.block4 = self.make_residual_layer(196, 256, 6)
        self.block5 = self.make_residual_layer(256, 384, 3)

    def make_residual_layer(self, input, output, num_layers):
        layer = []
        for i in range(num_layers):
            layer.append(nn.Conv2d(input, output // 4, kernel_size=3, stride=1, padding=1))
            layer.append(nn.BatchNorm2d(output // 4))
            layer.append(nn.ReLU(inplace=True))
            layer.append(nn.Conv2d(output, output // 4, kernel_size=3, stride=1, padding=1))
            layer.append(nn.BatchNorm2d(output // 4))
            layer.append(nn.ReLU(inplace=True))
            layer.append(nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1))
            layer.append(nn.BatchNorm2d(output))
            layer.append(nn.ReLU(inplace=True))
            if i == 0:
                layer[0] = nn.Conv2d(input, output // 4, kernel_size=3, stride=2, padding=1)
                layer.append(nn.Sequential(
                    nn.Conv2d(input, output // 4)
                ))



class Pixor(nn.Module):
    def __init__(self):
        super(Pixor, self).__init__()
        self.backbone = Backbone()
