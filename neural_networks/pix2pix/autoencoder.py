import torch
import visdom
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim
from dataset import MyDataSet
import argparse
import os
import time
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=80, help="interval between image sampling")
parser.add_argument("--dataset_name", type=str, default="ae1", help="name of the dataset")
parser.add_argument("--rfdata_len", type=int, default=1024, help="length of rf data")
parser.add_argument("--split_test", type=bool, default=True, help="if split test")
parser.add_argument("--network", type=str, default="ae", help="network")

opt = parser.parse_args()
print(opt)

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            # [b, 784] => [b, 256]
            nn.Linear(784, 256),
            nn.ReLU(),
            # [b, 256] => [b, 64]
            nn.Linear(256, 64),
            nn.ReLU(),
            # [b, 64] => [b, 20]
            nn.Linear(64, 20),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # [b, 20] => [b, 64]
            nn.Linear(20, 64),
            nn.ReLU(),
            # [b, 64] => [b, 256]
            nn.Linear(64, 256),
            nn.ReLU(),
            # [b, 256] => [b, 784]
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param [b, 1, 28, 28]:
        :return [b, 1, 28, 28]:
        """
        batchsz = x.size(0)
        # flatten
        x = x.view(batchsz, -1)
        # encode
        x = self.encoder(x)
        # decode
        x = self.decoder(x)
        # reshape
        x = x.view(batchsz, 1, 28, 28)

        return x

os.makedirs("images", exist_ok=True)
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)
dataloader = DataLoader(
    MyDataSet(root="./dataset/", opt=opt, img_transform=img_transforms, rf_transform=rf_transforms),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = DataLoader(
    MyDataSet(root="./dataset/", opt=opt, img_transform=img_transforms, rf_transform=rf_transforms, mode="test"),
    batch_size=10,
    shuffle=True,
    num_workers=1,
)
model = AE()
criteon = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
print(model)
prev_time = time.time()
viz = visdom.Visdom()

def sample_result():

    """Saves a generated sample from the validation set"""
    datas = next(iter(val_dataloader))

    rf_data = datas["rf_data"]

    result = model(rf_data)

    loss = criteon(result, rf_data)

    return loss

def TrainModelAE():
    for epoch in range(opt.n_epochs):
        for i, batch in enumerate(dataloader):
            rf_data = batch["rf_data"]
            x_hat = model(rf_data)
            loss = criteon(x_hat, rf_data)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s\r\n"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)


if __name__ == '__main__':
    TrainModelAE()