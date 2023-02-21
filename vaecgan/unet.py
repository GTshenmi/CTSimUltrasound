import torch.nn as nn
import torch

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [
            #nn.Conv2d(in_size, in_size, 3, 1, 1),
            nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.Conv2d(out_size, out_size, 3, 1, 1),
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
        layers = [
            #nn.ConvTranspose2d(in_size, in_size, 3, 1, 1, bias=False),
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(out_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x