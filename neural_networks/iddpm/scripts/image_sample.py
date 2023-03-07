"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys

import numpy as np
import torch as th
import torch.distributed as dist
from torchvision.utils import save_image
from PIL import Image

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main(use_gpu):
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    log_dir = f'/home/xuepeng/ultrasound/neural_networks/iddpm/run_record/test_{args.model_path.split("/")[-1].replace(".pt","")}'
    logger.configure(dir=log_dir)

    device = th.device("cuda:" + str(use_gpu))
    logger.log(args)
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    #model.to(dist_util.dev())
    model.to(device)
    model.eval()

    logger.log("sampling...")
    # all_images = []
    # all_labels = []
    sample_dir = os.path.join(logger.get_dir(), f'samples_{args.model_path.split("/")[-1].replace(".pt","")}/')
    # print(sample_dir)
    os.makedirs(sample_dir, exist_ok=True)
    index = 0
    while index * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        img_data = [sample.cpu().numpy() for sample in gathered_samples]
        # all_images.extend(img_data)
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        sample_path = os.path.join(sample_dir, f"samples_{index}.png")

        img_data = np.squeeze(img_data,0)
        img_data = np.squeeze(img_data,3)
        # img_data = np.transpose(img_data,[1,2,0])

        img_sample = img_data[0]
        for i in range(1,args.batch_size):
            img_sample = np.concatenate((img_sample,img_data[i]),axis=1)

        # logger.log(f'{np.shape(img_data)}')
        # logger.log(f'{np.shape(img_sample)}')
        # save_image(img_sample, sample_path, nrow=5, normalize=False)
        imgs = Image.fromarray(img_sample)
        imgs.save(sample_path)

        logger.log(f"created {index * args.batch_size} samples")
        index = index + 1

    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # if args.class_cond:
    #     label_arr = np.concatenate(all_labels, axis=0)
    #     label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     if args.class_cond:
    #         np.savez(out_path, arr, label_arr)
    #     else:
    #         np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1000,
        batch_size=2,
        use_ddim=False,
        model_path="/home/xuepeng/ultrasound/neural_networks/iddpm/run_record/openai-2023-03-03-17-23-23-938923/ema_0.9999_200000.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    use_gpu = 1
    with th.cuda.device(use_gpu):
        main(use_gpu)

