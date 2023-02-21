import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # b, 64, 128, 1024
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 128, 128, 1024
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor = 2,mode = "bilinear"),# b, 128, 256, 2048
            nn.MaxPool2d((1,2)),  # b, 128, 256, 1024

            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # b, 64, 256, 1024
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,8, 3, stride=1, padding=1),  # b, 4, 256, 1024
            nn.InstanceNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((1, 4)),  # b, 4, 256, 256
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(8, 64,(3,4), stride=(1,4), padding=(1,0)),  # b, 64, 256, 1024
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1),  # b, 128, 256, 1024
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, (3,4), stride=(1,2), padding=(1,1)),  # b, 128, 256, 2048
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor = 0.5,mode = "bilinear",recompute_scale_factor = True),# b, 128, 128, 1024

            nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1),  # b, 1, 128, 1024
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh(),
        )


    def forward(self, x):

        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc,dec
