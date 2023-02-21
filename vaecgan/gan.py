import torch
import torch.nn as nn
from unet import UNetUp,UNetDown

class Generator(nn.Module):
    def __init__(self, in_channels=8, out_channels=1,img_shape=(1, 256, 256)):
        super(Generator, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)#b 8 256 256 -> b 64 128 128
        self.down2 = UNetDown(64, 128)                         #b 64 128 128 -> b 128 64 64
        self.down3 = UNetDown(128, 256,dropout=0.5)            #b 128 64 64 -> b 256 32 32
        self.down4 = UNetDown(256, 512, dropout=0.5)           #b 256 32 32 -> b 512 16 16
        self.down5 = UNetDown(512, 512)                        #b 512 16 16 -> b 512 8 8

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 1024,3, 1, 1),  # b 512 8 8 -> b 1024 8 8
            nn.ConvTranspose2d(1024, 1024, 3,1,1),  # b 1024 8 8 -> b 1024 8 8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1,bias = False), # b, 1024, 8, 8 -> b 512 16 16
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up2 = UNetUp(1024, 256, dropout=0.5) # b 512+512 16 16 -> b 256 32 32
        self.up3 = UNetUp(512, 128, dropout=0.5) # b 256+256 32 32 -> b 128 64 64
        self.up4 = UNetUp(256, 64) # b 128+128 64 64 -> b 64 128 128
        self.up5 = UNetUp(128, 32)  # b 64+64 128 128 -> b 32 256 256

        self.final = nn.Sequential(
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.Conv2d(16, out_channels, 3, 1,1),
            nn.Tanh(),
        )

        self.img_shape = img_shape

    def forward(self,condition):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(condition)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        img = self.final(u5)

        img = img.view(img.size(0), *self.img_shape)

        return img

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_filters, in_filters, 3, stride=1, padding=1),
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1),
            ]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 9, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self,condition,img):
        # Concatenate image and condition image by channels to produce input

        img_input = torch.cat((img,condition), 1)

        outpuut = self.model(img_input)

        return outpuut
