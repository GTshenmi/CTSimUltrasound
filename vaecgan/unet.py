import torch.nn as nn
import torch

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 3, 1, 1),
            nn.Conv2d(out_size, out_size, 3, 1, 1),
            nn.Conv2d(out_size, out_size, 4, 2, 1, bias=False),
                  ]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()

        mid_size = int(in_size/2)

        layers = [
            nn.ConvTranspose2d(in_size, mid_size, 3, 1, 1),
            nn.ConvTranspose2d(mid_size, mid_size, 3, 1, 1),
            nn.ConvTranspose2d(mid_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2)
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):

        x = torch.cat((x, skip_input), 1)
        x = self.model(x)

        return x