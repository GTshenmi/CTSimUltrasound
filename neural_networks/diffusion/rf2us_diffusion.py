from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
import torch
from autoencoder import AutoEncoder
from datasetnew import MyDataSet

import os
import numpy as np
import time
import datetime
import sys
import argparse

from itertools import chain
import torch as th
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from improved_diffusion import dist_util, logger
from improved_diffusion.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from improved_diffusion.nn import update_ema
from improved_diffusion.resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0
class AEDiffusion():
    def __init__(self,args):

        self.cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.device = torch.device("cuda:" + str(args.use_gpu) if self.cuda else "cpu")

        dist_util.setup_dist()

        self.img_shape = (args.channels, args.image_size, args.image_size)
        self.input_shape = (128, args.rfdata_len)

        os.makedirs("images/%s" % args.train_name, exist_ok=True)
        os.makedirs("saved_models/%s" % args.train_name, exist_ok=True)
        os.makedirs("loss/%s" % args.train_name, exist_ok=True)

        self.dataloader = DataLoader(
            MyDataSet(root="../datasetnew/",args = args),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_cpu,
            drop_last = True,
        )

        self.val_dataloader = DataLoader(
            MyDataSet(root="../datasetnew/",args = args,mode="test"),
            batch_size=2,
            shuffle=True,
            num_workers=1,
        )

        # Initialize generator and discriminator
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        self.model.to(self.device)
        self.schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, self.diffusion)

        self.autoencoder = AutoEncoder(self.device)

        self.batch_size = args.batch_size
        self.microbatch = args.microbatch if args.microbatch > 0 else args.batch_size
        self.lr = args.lr
        self.ema_rate = (
            [args.ema_rate]
            if isinstance(args.ema_rate, float)
            else [float(x) for x in args.ema_rate.split(",")]
        )
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = args.use_fp16
        self.fp16_scale_growth = args.fp16_scale_growth

        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(chain(self.model.parameters(),self.autoencoder.parameters()))
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()

        if self.use_fp16:
            self._setup_fp16()

        self.optimizer = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available() and True:
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.device],
                output_device=self.device,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            # print("not use ddp")
            self.use_ddp = False
            self.ddp_model = self.model

        self.args = args

        self.loss = []

        self.train_loss = np.zeros(5)
        self.test_loss = np.zeros(5)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()
    def _load_and_sync_parameters(self):
        pass

    def _load_ema_parameters(self, rate):
        pass

    def _load_optimizer_state(self):
        pass
    def forward_backward(self, batch, cond):

        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):

            micro_us = batch[i : i + self.microbatch].to(self.device)

            micro_rf = cond[i : i + self.microbatch].to(self.device)

            last_batch = (i + self.microbatch) >= batch.shape[0]

            t, weights = self.schedule_sampler.sample(micro_us.shape[0], self.device)

            ae_out = self.autoencoder.training_losses(micro_rf)

            # print((micro_us.shape,micro_rf.shape,ae_out["enc"].shape))

            model_kwargs = {
                "encoder_rf": ae_out["enc"],
            }


            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro_us,
                t,
                model_kwargs=model_kwargs,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean() + ae_out["loss"]

            # log_loss_dict(
            #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
            # )

            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

            # self.loss = loss
            # self.ae_loss = ae_out["loss"]
            self.train_loss = {
                "loss":loss,
                "ae_loss":ae_out["loss"],
            }


    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        # self._log_grad_norm()
        self._anneal_lr()
        self.optimizer.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.args.param_groups:
            param_group["lr"] = lr
    def optimize_normal(self):
        # self._log_grad_norm()
        self._anneal_lr()
        self.optimizer.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def TrainModel(self):

        prev_time = time.time()

        # print(self.model)

        for epoch in range(self.args.resume_epochs,self.args.epochs + self.args.resume_epochs):
            # train_loss = np.zeros(5)
            for i, batch in enumerate(self.dataloader):

                # Model inputs
                us_images = batch["us"].type(self.Tensor)
                rf_datas = batch["rf_data"].type(self.Tensor)

                # -----------------
                #  Train Generator And AutoEncoder
                # -----------------
                self.forward_backward(us_images, rf_datas)

                if self.use_fp16:
                    self.optimize_fp16()
                else:
                    self.optimize_normal()

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(self.dataloader) + i
                batches_left = (self.args.epochs + self.args.resume_epochs) * len(self.dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Diffusion loss: %f] [AE loss: %f] ETA: %s\r\n"
                    % (
                        epoch,
                        self.args.epochs + self.args.resume_epochs,
                        i,
                        len(self.dataloader),
                        self.train_loss["loss"],
                        self.train_loss["ae_loss"],
                        time_left,
                    )
                )

        #         current_loss_list = []
        #
        #         current_loss_list.append(loss_D.item())
        #         current_loss_list.append(loss_G.item())
        #         current_loss_list.append(loss_AE.item())
        #         current_loss_list.append(loss_pixel.item())
        #         current_loss_list.append(loss_GAN.item())
        #
        #         current_loss = np.array(current_loss_list)
        #
        #         train_loss = train_loss + current_loss
        #
        #     train_loss = train_loss/len(self.dataloader)
        #
        #     self.train_loss = np.vstack((self.train_loss,train_loss))
        #
        #     #self.CalTestLoss()
        #
        #     self.SampleImg(epoch,real_imgs,fake_imgs)
        #
        #     self.SaveModel(epoch)
        #
        #     np.save("loss/%s/train_loss.npy" % (opt.dataset_name), self.train_loss)
        #
        # np.save("loss/%s/train_loss.npy" % (opt.dataset_name), self.train_loss)

    def SampleImg(self,batches_done,train_real,train_fake):
        pass
    def SaveModel(self):
        pass


def create_argparser():
    defaults = dict(
        data_dir="/home/xuepeng/ultrasound/neural_networks/datasetnew/us_image/",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        epochs = 200,
        use_gpu = 1,
        channels = 1,
        train_name = "rf2us0",
        n_cpu = 6,
        rfdata_len = 1024,
        resume_epochs = 0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == '__main__':
    os.makedirs("../images", exist_ok=True)

    args = create_argparser().parse_args()

    print(args)

    torch.cuda.set_device(args.use_gpu)

    with torch.cuda.device(args.use_gpu):
        diffusion = AEDiffusion(args)
        diffusion.TrainModel()



