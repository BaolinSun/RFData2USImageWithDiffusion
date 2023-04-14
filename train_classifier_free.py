import os
import argparse
import datetime
import torch

import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from improved_diffusion.resample import create_named_schedule_sampler
from pprint import pprint
from utils.diffusion import (
    create_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from datasets import (
    PicmusTrainDataset,
    PicmusValDataset,
    RFDataUsImageTrain,
    RFDataUsImageValidation,)
from utils.logger import create_logger
from autoencoder import AutoEncoder
from itertools import chain
from util import save_image


def main():
    args = create_argparser().parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + '_' + args.message
    logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    imgdir = os.path.join(logdir, "images/val")
    os.makedirs(logdir)
    os.mkdir(ckptdir)
    os.makedirs(imgdir)
    logger = create_logger(output_dir=logdir, name="")

    logger.info(args)
    device = (torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu"))

    logger.info("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    autoencoder = AutoEncoder()
    model.to(device)
    autoencoder.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)


    optimizer = torch.optim.AdamW(params=chain(model.parameters(),autoencoder.parameters()), lr = args.lr, weight_decay=args.weight_decay)

    criterion_AE = torch.nn.MSELoss()
    criterion_AE.to(device)

    # configure dataloder
    logger.info("creating data loader...")
    rf_transforms = [
        transforms.ToTensor(),
    ]
    us_transforms = [
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
    ]

    training_images_list_file = 'some/us_image_train.txt'
    test_images_list_file = 'some/us_image_val.txt'
    # training_images_list_file = 'picmus/picmus_train.txt'
    # test_images_list_file = 'picmus/picmus_val.txt'
    train_dataloader = DataLoader(
        RFDataUsImageTrain(training_images_list_file=training_images_list_file, us_transforms=us_transforms, rf_transforms=rf_transforms),
        # PicmusTrainDataset(root='picmus', train_list_file=training_images_list_file, us_transforms=self.us_transforms, rf_transforms=self.rf_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        drop_last = True,
    )

    val_dataloader = DataLoader(
        RFDataUsImageValidation(test_images_list_file=test_images_list_file, us_transforms=us_transforms, rf_transforms=rf_transforms),
        # PicmusValDataset(root='picmus', test_list_file=test_images_list_file, us_transforms=self.us_transforms, rf_transforms=self.rf_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
    )
    dataloader = val_dataloader

    g_step = 0
    writer = SummaryWriter(logdir)
    for epoch in range(args.n_epoch):
        model.train()
        for i, batch in enumerate(train_dataloader):

            # Model inputs
            us_image = batch["us_image"].to(device)
            rf_data = batch["rf_data"].to(device)

            encoder_rf,decoder_rf = autoencoder(rf_data)

            t, weights = schedule_sampler.sample(us_image.shape[0], device)

            optimizer.zero_grad()

            # Autoencoder loss
            loss_ae = criterion_AE(decoder_rf, rf_data)

            # Diffusion loss
            losses = diffusion.training_losses(model=model, x_start=us_image, t=t, context=encoder_rf)
            loss_unet = (losses["loss"] * weights).mean()

            loss = loss_unet + 0.2*loss_ae

            loss.backward()
            optimizer.step()

            writer.add_scalar(tag='loss/train', scalar_value=loss, global_step=g_step)
            writer.add_scalar(tag='loss/unet', scalar_value=loss_unet, global_step=g_step)
            writer.add_scalar(tag='loss/ae', scalar_value=loss_ae, global_step=g_step)

            if (g_step + 1) % args.log_interval == 0:
                logger.info(f"epoch: {epoch}, step: {g_step+1},  loss: {loss.item()}, loss_unet: {loss_unet.item()}, loss_ae: {loss_ae.item()}")

            if (g_step + 1) % args.sample_interval == 0:
                model.eval()
                autoencoder.eval()
                with torch.no_grad():
                    model_kwargs = {}
                    sample_fn = (
                        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                    )
                    sample = sample_fn(model, (args.batch_size, 3, args.image_size, args.image_size), context=encoder_rf,
                                        clip_denoised=args.clip_denoised, progress=args.progress, model_kwargs=model_kwargs)
                    
                    res = torch.cat([sample, us_image], dim=0)      
                    save_image(res, os.path.join(imgdir, f'sample_epoch{epoch}_step{g_step}_train.png'), nrow = args.batch_size)

                    try:
                        batch = next(iter(dataloader))
                    except StopIteration:
                        dataloader = val_dataloader
                        batch = next(iter(dataloader))
                    us_image = batch["us_image"].to(device)
                    rf_data = batch["rf_data"].to(device)
                    encoder_rf = autoencoder.rf_encoder(rf_data)
                    sample = sample_fn(model, (args.batch_size, 3, args.image_size, args.image_size), context=encoder_rf,
                                        clip_denoised=args.clip_denoised, progress=args.progress, model_kwargs=model_kwargs)
                    res = torch.cat([sample, us_image], dim=0)      
                    save_image(res, os.path.join(imgdir, f'sample_epoch{epoch}_step{g_step}_eval.png'), nrow = args.batch_size)

            g_step += 1



if __name__ == "__main__":
    main()