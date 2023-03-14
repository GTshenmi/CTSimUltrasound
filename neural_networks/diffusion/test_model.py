from rf2us_diffusion import AEDiffusion
import argparse
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch
import os

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
        use_gpu = 0,
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
        diffusion.TestModel()