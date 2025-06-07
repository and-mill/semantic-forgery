"""
Script to run the reprompting attack.
"""


import os

import torch

import pandas as pd

import argparse

from utils.imprint_utils import validate
from utils.wm.wm_utils import WmProviders
from utils.wm.gs_provider import parser as gs_parser
from utils.wm.tr_provider import parser as tr_parser

from utils.utils import get_detection_threshold, check_if_detection_successful

from utils.pipe import pipe_utils

from utils.prompt_utils import PROMPTS_SD_LIST, PROMPTS_I2P_LIST

from utils.utils import set_random_seed


# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# args
parser = argparse.ArgumentParser(description="reprompt", parents=[gs_parser, tr_parser])

parser.add_argument("--out_dir", type=str, default="out/reprompt/")

# prompts
parser.add_argument("--target_prompt_index", type=int, default=0, choices=list(range(len(PROMPTS_SD_LIST))))
parser.add_argument("--target_prompt", type=str, default=None)
parser.add_argument("--attacker_prompt_index", type=int, default=0, choices=list(range(len(PROMPTS_I2P_LIST))))
parser.add_argument("--attacker_prompt", type=str, default=None)

# target model
parser.add_argument("--modelid_target",
                    type=str,
                    default="stabilityai/stable-diffusion-xl-base-1.0",
                    choices=["stabilityai/stable-diffusion-xl-base-1.0", "PixArt-alpha/PixArt-Sigma-XL-2-512-MS", "black-forest-labs/FLUX.1-dev"])
parser.add_argument("--scheduler_target", type=str, default="DDIM")
parser.add_argument("--guidance_scale_target", type=float, default=7.5)  # 20 for FLUX
parser.add_argument("--num_inference_steps_target", type=int, default=50)  # 3.5 for FLUX

# attacker model
parser.add_argument("--modelid_attacker", type=str, default="stabilityai/stable-diffusion-2-1-base")
parser.add_argument("--scheduler_attacker", type=str, default="DDIM")
parser.add_argument("--num_inference_steps_attacker", type=int, default=50)
parser.add_argument("--guidance_scale_attacker", type=float, default=7.5)

parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--wm_type",
                    type=str,
                    default="GS",
                    choices=[wm.name for wm in WmProviders])
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--resample", action="store_true", default=False)

args = parser.parse_args()

# save outputin a subfolder defined by the index of the prompt we're using if explicit prompt is not given
out_dir = os.path.join(args.out_dir,
                       f"target_prompt_index={args.target_prompt_index if args.target_prompt is None else 'custom'}",
                       f"attacker_prompt_index={args.attacker_prompt_index if args.attacker_prompt is None else 'custom'}")

# prompts are taken from a predefined list of SD-Prompts (https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)
target_prompt = PROMPTS_SD_LIST[args.target_prompt_index] if args.target_prompt is None else args.target_prompt
# prompts are taken from a predefined list of I2P (https://huggingface.co/datasets/AIML-TUDA/i2p)
attacker_prompt = PROMPTS_I2P_LIST[args.attacker_prompt_index] if args.attacker_prompt is None else args.attacker_prompt
# add full prompt datasets here if you like

# attacker model
pipe_provider_target = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_target,
                                                    resolution=args.resolution,
                                                    schedulers_name=args.scheduler_target,
                                                    unet_id_or_checkpoint_dir=None,
                                                    lora_checkpoint_dir=None,
                                                    device=DEVICE,
                                                    eager_loading=True if "FLUX" in args.modelid_target else False,
                                                    disable_tqdm=True
                                                    )  # finetuned model
pipe_provider_attacker = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_attacker,
                                                      resolution=args.resolution,
                                                      device=DEVICE,
                                                      eager_loading=False,
                                                      disable_tqdm=True
                                                      )  # base model

# set seeds
set_random_seed(args.seed)

# retrieve the detection threshold for the settings
detection_threshold = get_detection_threshold(args.wm_type, args.modelid_target)

rows = []
with torch.no_grad():
            
    # --------------------------------------------------------------- PHASE 1 ----------------------------------------------------------------------
    print("phase 1: generate target image")

    # generate a watermarked latent zT
    wm_provider = WmProviders[args.wm_type].value(latent_shape=pipe_provider_target.get_latent_shape(), **vars(args))
    wm_initial_results = wm_provider.get_wm_latents()
    wm_zT = wm_initial_results["zT_torch"]

    # generate a benign image
    res_1 = pipe_provider_target.generate(prompts=target_prompt,
                                          num_inference_steps=args.num_inference_steps_target,
                                          guidance_scale=args.guidance_scale_target,
                                          latents=wm_zT)
    benign_image = res_1["images_PIL"][0]
        
    # for Gaussian Shading, we also get an initial message
    message_bits_str_initial = wm_initial_results["message_bits_str_list"][0] if "message_bits_str_list" in wm_initial_results else None

    # collect metrics
    results = validate(
        out_dir=out_dir,
        image_to_verify_PIL=benign_image,
        original_PIL=benign_image,
        wm_provider=wm_provider,
        pipe_provider_target=pipe_provider_target,
        num_inference_steps_target=args.num_inference_steps_target,
        step=-1,
        message_bits_str_initial=message_bits_str_initial,
        do_psnr=False,
        do_ssim=False,
        do_msssim=False,
        do_lpips=False,
        )
    # check if detection was successfull
    detection_successful = check_if_detection_successful(wm_type=args.wm_type,
                                                         threshold=detection_threshold,
                                                         value=results["bit_accuracy"] if args.wm_type == "GS" else results["p_value"])
    results["detection_successful"] = detection_successful
    rows.append(results)

    # log
    print(f"(Benign image) detection_success: {detection_successful}, bit accuracy: {results['bit_accuracy']:.5f}, p_value: {results['p_value']}")
    
    # --------------------------------------------------------------- PHASE 2 ----------------------------------------------------------------------
    print("phase 2: invert using attacker model")
    
    pipe_provider_target.stash_pipe()
    res_2 = pipe_provider_attacker.invert_images(images=res_1["images_torch"], num_inference_steps=args.num_inference_steps_attacker)

    # --------------------------------------------------------------- PHASE 3 ----------------------------------------------------------------------
    print("phase 3: generate attacker image")

    # resample strategy, used in Reprompt+ attack, but only for GS
    # For TR, there is no resampling, but merely trying out multiple attacker prompts and choosing the best performing sample
    if args.resample:
        recovered_zT = wm_provider.wiggle_latents(res_2["zT_torch"].clone())
        recovered_zT = recovered_zT.to(dtype=pipe_provider_attacker.get_dtype())
    else:
        recovered_zT = res_2["zT_torch"].clone()

    # generate a harmful image
    res_3 = pipe_provider_attacker.generate(prompts=attacker_prompt,
                                            num_inference_steps=args.num_inference_steps_attacker,
                                            guidance_scale=args.guidance_scale_attacker,
                                            latents=recovered_zT,
                                            )
    harmful_image = res_3["images_PIL"][0]
    
    # --------------------------------------------------------------- PHASE 4 ----------------------------------------------------------------------
    print("phase 4: invert using target model and verify watermark")

    pipe_provider_attacker.stash_pipe()
    
    # collect metrics
    results = validate(
        out_dir=out_dir,
        image_to_verify_PIL=harmful_image,
        original_PIL=benign_image,
        wm_provider=wm_provider,
        pipe_provider_target=pipe_provider_target,
        num_inference_steps_target=args.num_inference_steps_target,
        step=1,
        message_bits_str_initial=message_bits_str_initial,
        do_psnr=False,
        do_ssim=False,
        do_msssim=False,
        do_lpips=False,
        )
    # check if detection was successfull
    detection_successful = check_if_detection_successful(wm_type=args.wm_type,
                                                         threshold=detection_threshold,
                                                         value=results["bit_accuracy"] if args.wm_type == "GS" else results["p_value"])
    results["detection_successful"] = detection_successful
    rows.append(results)

    # log
    print(f"(Harmful image) detection_success: {detection_successful}, bit accuracy: {results['bit_accuracy']:.5f}, p_value: {results['p_value']}")

    # save metrics as csv
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"metrics.csv"))
