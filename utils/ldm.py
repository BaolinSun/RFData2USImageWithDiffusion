import os
import argparse

from improved_diffusion.openaimodel import UNetModel
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
from improved_diffusion import gaussian_diffusion as gd


NUM_CLASSES = 1000


def create_argparser():
    defaults = dict(
        data_dir="/home/sbl/datasets/some/us_image",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=1000,
        sample_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_ddim = False,
        progress=True,
        clip_denoised=True,
        n_epoch = 10000,
        n_cpu = 8,
        gpu = 0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("-m",
                    "--message",
                    help="training info",
                    type=str,
                    default='Null')

    return parser


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size = 64,
        in_channels = 3,
        out_channels = 3,
        model_channels = 128,
        attention_resolutions = [4, 2, 1],
        num_res_blocks=2,
        channel_mult = [1, 2, 4],
        num_head_channels = 32,
        use_spatial_transformer=False,
        transformer_depth= 1,
        context_dim=None,
        num_channels=256,
        num_heads=4,
        num_heads_upsample=-1,
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_model_and_diffusion(
    image_size,
    in_channels,
    out_channels,
    model_channels,
    attention_resolutions,
    num_res_blocks,
    channel_mult,
    num_head_channels,
    use_spatial_transformer,
    transformer_depth,
    context_dim,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_heads,
    num_heads_upsample,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = create_model(
    image_size,
    in_channels,
    out_channels,
    model_channels,
    attention_resolutions,
    num_res_blocks,
    channel_mult,
    num_head_channels,
    use_spatial_transformer,
    transformer_depth,
    context_dim
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )

    return model, diffusion


def create_model(
    image_size= 32,
    in_channels= 4,
    out_channels= 4,
    model_channels= 256,
    attention_resolutions = [4, 2, 1],
    num_res_blocks= 2,
    channel_mult = [1, 2, 4],
    num_head_channels= 32,
    use_spatial_transformer=False,
    transformer_depth= 1,
    context_dim=None
):

    return UNetModel(image_size=image_size, 
                     in_channels=in_channels,
                     out_channels=out_channels,
                     model_channels=model_channels,
                     attention_resolutions=attention_resolutions,
                     num_res_blocks=num_res_blocks,
                     channel_mult=channel_mult,
                     num_head_channels=num_head_channels,
                     use_spatial_transformer=use_spatial_transformer,
                     transformer_depth=transformer_depth,
                     context_dim=context_dim)


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):

    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    use_timesteps = space_timesteps(steps, timestep_respacing)
    model_mean_type = (
        gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
    )
    model_var_type = (
        (
            gd.ModelVarType.FIXED_LARGE
            if not sigma_small
            else gd.ModelVarType.FIXED_SMALL
        )
        if not learn_sigma
        else gd.ModelVarType.LEARNED_RANGE
    )

    return SpacedDiffusion(
        use_timesteps=use_timesteps,
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type = model_var_type,
        loss_type = loss_type,
        rescale_timesteps = rescale_timesteps
    )


# ====================================================================


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
    


