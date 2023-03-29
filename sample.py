import os
import torch
import argparse

from util import save_image

from utils.diffusion import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()

    device = (torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu"))

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print("sampling...")

    all_images = []
    # model_kwargs = {}
    # while len(all_images) * args.batch_size < args.num_samples:
    model_kwargs = {}
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    sample = sample_fn(model, (args.batch_size, 3, args.image_size, args.image_size),
                        clip_denoised=args.clip_denoised, progress=args.progress, model_kwargs=model_kwargs)
    # sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # sample = sample.permute(0, 2, 3, 1)
    # sample = sample.contiguous()

    save_image(sample, 'test.png')







def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=8,
        use_ddim=False,
        progress=True,
        model_path="logs/2023-03-28T09-39-51/checkpoints/model045799.pth",
        gpu = 0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == '__main__':
    main()