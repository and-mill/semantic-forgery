"""
Script to run the removal attack.
"""

import os

import PIL
import PIL.Image
import torch

import pandas as pd

import tqdm

import argparse

from utils.wm.wm_utils import WmProviders
from utils.wm.gs_provider import parser as gs_parser
from utils.wm.tr_provider import parser as tr_parser

from utils import imprint_utils
from utils.imprint_utils import invert_image, validate
from utils.utils import get_detection_threshold, check_if_detection_successful

from utils.pipe import pipe_utils

from utils.prompt_utils import PROMPTS_SD_LIST

from utils.utils import set_random_seed

import lpips



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# args
parser = argparse.ArgumentParser(description="imprint-removal", parents=[gs_parser, tr_parser])

parser.add_argument("--out_dir", type=str, default="out/imprint-removal/")

parser.add_argument("--target_prompt_index", type=int, default=0, choices=list(range(len(PROMPTS_SD_LIST))))
parser.add_argument("--target_prompt", type=str, default=None)

# target model
parser.add_argument("--modelid_target",
                    type=str,
                    default="stabilityai/stable-diffusion-xl-base-1.0",
                    choices=["stabilityai/stable-diffusion-xl-base-1.0", "PixArt-alpha/PixArt-Sigma-XL-2-512-MS", "black-forest-labs/FLUX.1-dev"])
parser.add_argument("--scheduler_target", type=str, default="DDIM")
parser.add_argument("--num_inference_steps_target", type=int, default=50)  # 3.5 for FLUX
parser.add_argument("--guidance_scale_target", type=float, default=7.5)  # 20 for FLUX

# attacker model
parser.add_argument("--modelid_attacker", type=str, default="stabilityai/stable-diffusion-2-1-base")
parser.add_argument("--scheduler_attacker", type=str, default="DDIM")
parser.add_argument("--num_inference_steps_attacker", type=int, default=50)

parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--wm_type",
                    type=str,
                    default="GS",
                    choices=[wm.name for wm in WmProviders])
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--steps", type=int, default=151)
parser.add_argument("--validation_steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=1)

args, unknown_args = parser.parse_known_args()

# set seeds
set_random_seed(args.seed)

# set up lpips
lpips_loss_fn = lpips.LPIPS(net='alex').to(device=DEVICE)

# retrieve the detection threshold for the settings
detection_threshold = get_detection_threshold(args.wm_type, args.modelid_target)

# save outputin a subfolder defined by the index of the prompt we're using if explicit prompt is not given
out_dir = os.path.join(args.out_dir, f"target_prompt_index={args.target_prompt_index if args.target_prompt is None else 'custom'}")

# prompts are taken from a predefined list of SD-Prompts (https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)
target_prompt = PROMPTS_SD_LIST[args.target_prompt_index] if args.target_prompt is None else args.target_prompt

# The pipe used by the attacker (SD2.1)
# For imprinting-type attacks, we do not use our usual pipe-wrappers on the attacker' side because the differentiable pipe requires a few extra steps that made it difficult to merge it with them.
# We might integrate this in the future, but for now, we use the pipe directly.
pipe_attacker, forward_scheduler, inverse_scheduler = imprint_utils.load_pipe(modelid=args.modelid_attacker,
                                                                              scheduler=args.scheduler_attacker,
                                                                              device=DEVICE)
# differentiable helper pipe used for propagating gradients through the inversion process
diffpipe = imprint_utils.DiffPipe(pipe_attacker, scheduler=inverse_scheduler, device=pipe_attacker.device)

# pipe_provider used by the target model (SDXL, PixArt, FLUX)
pipe_provider_target = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_target,
                                                    resolution=args.resolution,
                                                    device=DEVICE,
                                                    eager_loading=True if "FLUX" in args.modelid_target else False,
                                                    disable_tqdm=True,)
    
# generate a watermarked latent zT
# This way like it is done here is a simple way to obtain a watermark provider for a simple test run.
# If you want to do mass experiments and have batch_sizes > 1, plz have look at the utils.wm_provider.WmProvider.generate_providers method
wm_provider = WmProviders[args.wm_type].value(latent_shape=pipe_provider_target.get_latent_shape(), **vars(args))
wm_initial_results = wm_provider.get_wm_latents()
wm_zT = wm_initial_results["zT_torch"]
# for Gaussian Shading, we also get an initial message
message_bits_str_initial = wm_initial_results["message_bits_str_list"][0] if "message_bits_str_list" in wm_initial_results else None

# generate a watermarked image with the target model
generated_PIL_list = pipe_provider_target.generate(prompts=target_prompt,
                                                   latents=wm_zT,
                                                   num_inference_steps=args.num_inference_steps_target,
                                                   guidance_scale=args.guidance_scale_target)["images_PIL"]
generated_PIL = generated_PIL_list[0]
generated_pt = pipe_provider_target.PIL_to_torch(generated_PIL_list)

# set up optimization
z0_original = imprint_utils.pixel_to_latent(generated_PIL, pipe_attacker)
z0 = torch.nn.Parameter(z0_original.detach().clone())
optim = torch.optim.Adam([z0], lr=args.lr)

# invert the watermarked image with the attacker model to get the target latent zT
with torch.no_grad():
    zT_retrieved = invert_image(pipe=pipe_attacker,
                                image_pt=generated_pt.to(dtype=torch.float32),
                                scheduler=inverse_scheduler,
                                num_inference_steps=args.num_inference_steps_attacker)
    zT_retrieved = zT_retrieved.detach() * -1  # the objective is flipped for removal

generated_PIL = generated_PIL[0] if isinstance(generated_PIL, list) else generated_PIL

# start to collect metrics on the cover image
rows = []
results = validate(
     out_dir=out_dir,
     image_to_verify_PIL=generated_PIL,
     original_PIL=generated_PIL,
     wm_provider=wm_provider,
     pipe_provider_target=pipe_provider_target,
     num_inference_steps_target=args.num_inference_steps_target,
     step=-1,
     message_bits_str_initial=message_bits_str_initial,
     lpips_loss_fn=lpips_loss_fn,
     device=DEVICE,
     )
# check if detection was successfull
detection_successful = check_if_detection_successful(wm_type=args.wm_type,
                                                     threshold=detection_threshold,
                                                     value=results["bit_accuracy"] if args.wm_type == "GS" else results["p_value"])
results["detection_successful"] = detection_successful
rows.append(results)

# log
print(
    f"Step {results['step']}, "
    f"detection_success: {detection_successful}, "
    f"bit accuracy: {results['bit_accuracy']:.5f}, "
    f"p_value: {results['p_value']}, "
    f"psnr: {results['psnr']:.5f}, "
    f"ssim: {results['ssim']:.5f}, "
    f"ms-ssim: {results['msssim']:.5f}, "
    f"lpips: {results['lpips']:.5f}"
)

# training loop
inverted_history = []
loss_history = []
for step in tqdm.tqdm(range(args.steps)):
    optim.zero_grad()
        
    # recons = diffpipe(zT, prompt)
    inverted_latent = diffpipe(z0, "", guidance_scale=1.)
        
    # get loss in zT space
    l = torch.nn.functional.mse_loss(inverted_latent, zT_retrieved)

    # do update
    l.backward()
    optim.step()
        
    # validate
    if step % args.validation_steps == 0:
        # get back to pixel space and send the attack instance to the target model for validation
        image_from_z0 = imprint_utils.latent_to_pil(z0, pipe_attacker)[0]

        # collect metrics on the imprinted image
        results = validate(
            out_dir=out_dir,
            image_to_verify_PIL=image_from_z0,
            original_PIL=generated_PIL,
            wm_provider=wm_provider,
            pipe_provider_target=pipe_provider_target,
            num_inference_steps_target=args.num_inference_steps_target,
            step=step,
            message_bits_str_initial=message_bits_str_initial,
            lpips_loss_fn=lpips_loss_fn,
            device=DEVICE,
            )
        # check if detection was successfull
        detection_successful = check_if_detection_successful(wm_type=args.wm_type,
                                                             threshold=detection_threshold,
                                                             value=results["bit_accuracy"] if args.wm_type == "GS" else results["p_value"])
        results["detection_successful"] = detection_successful
        rows.append(results)

        # log
        print(
            f"Step {results['step']}, "
            f"detection_success: {detection_successful}, "
            f"bit accuracy: {results['bit_accuracy']:.5f}, "
            f"p_value: {results['p_value']}, "
            f"psnr: {results['psnr']:.5f}, "
            f"ssim: {results['ssim']:.5f}, "
            f"ms-ssim: {results['msssim']:.5f}, "
            f"lpips: {results['lpips']:.5f}"
        )

        # save metrics as csv every validation round
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, f"metrics.csv"))
