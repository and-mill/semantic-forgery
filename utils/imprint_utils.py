# This source file contains code for the pipes that the attacker uses during imprinting attacks.
# It required implementing a differentiable pipe with gradient checkpointing.

import typing

import os

from typing import List, Optional, Union
import PIL
import PIL.Image
import torch

from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler, DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg

from torchvision.transforms.functional import to_pil_image, to_tensor

from torch.utils.checkpoint import checkpoint

from utils.image_utils import psnr_PIL, ssim_PIL, msssim_PIL, lpips_PIL
from utils.wm.wm_provider import WmProvider


def load_pipe(modelid="stabilityai/stable-diffusion-2-1-base", scheduler="DDIM", device="cuda") -> typing.Tuple[StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler]:
    """
    Simple util for loading a attacker pipe and the inverse scheduler which we need later for inversion.

    @param modelid: str, the model id
    @param scheduler: str, the scheduler
    @param device: str, the device

    @return: tuple, the pipe, forward scheduler, inverse scheduler
    """

    modelid = "stabilityai/stable-diffusion-2-1-base" if modelid is None else modelid
    if scheduler == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(modelid, subfolder='scheduler', torch_dtype=torch.float32)
        inverse_scheduler = DDIMInverseScheduler.from_pretrained(modelid, subfolder='scheduler', torch_dtype=torch.float32)
    elif scheduler == "DPM":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(modelid, subfolder='scheduler', torch_dtype=torch.float32)
        inverse_scheduler = DPMSolverMultistepInverseScheduler.from_pretrained(modelid, subfolder='scheduler', torch_dtype=torch.float32)
    else:
        raise ValueError(f"Unknown attacker scheduler: {scheduler}")

    # Load the pre-trained Stable Diffusion 2.1 model from Hugging Face
    pipe = StableDiffusionPipeline.from_pretrained(modelid,
                                                   scheduler=scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=torch.float32).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    return pipe, scheduler, inverse_scheduler


class DiffPipe(torch.nn.Module):
    """
    A class for backpropagating through a pipeline's generation loop using gradient checkpointing.
    Works well for SD2.1
    """
    def __init__(self, pipe, scheduler=None, device=None):
        super().__init__()
        self.pipe = pipe
        self.device = device
        self.scheduler = scheduler
        
        self.unet = UnetWrapper(self.pipe.unet)
        
        for param in self.pipe.unet.parameters():
            param.requires_grad = False
        
    def forward(self, 
                latents=None,
                prompt="", 
                negative_prompt="",
                prompt_embeds=None,
                guidance_scale=7.5,
                num_inference_steps: int = 50,
                device=None,
                do_classifier_free_guidance=None,
                generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                eta: float = 0.0,
                ):
        """
        Forward pass through the pipeline with gradient checkpointing.

        @param latents: torch.Tensor, the latents
        @param prompt: str, the prompt
        @param negative_prompt: str, the negative prompt
        @param prompt_embeds: torch.Tensor, the prompt embeddings
        @param guidance_scale: float, the guidance scale
        @param num_inference_steps: int, the number of inference steps
        @param device: torch.device, the device
        @param do_classifier_free_guidance: bool, whether to do classifier free guidance
        @param generator: torch.Generator, the generator
        @param eta: float, the eta
        """
        oldsched = self.pipe.scheduler
        self.pipe.scheduler = self.scheduler
        
        sigmas = None
        batch_size = 1
        num_images_per_prompt = 1
        timesteps = None
        do_classifier_free_guidance = (guidance_scale > 1.) if do_classifier_free_guidance is None else do_classifier_free_guidance
        
        device = device if device is not None else self.device
        
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                guidance_scale > 0,
                negative_prompt
            )
            
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        else:
            prompt_embeds = prompt_embeds
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        latents = latents * self.scheduler.init_noise_sigma

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = None

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.pipe._num_timesteps = len(timesteps)
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipe.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = checkpoint(self.unet, 
                    latent_model_input,
                    t,
                    prompt_embeds,
                    timestep_cond,
                    self.pipe.cross_attention_kwargs,
                    added_cond_kwargs,
                    False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and self.pipe.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.pipe.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self.pipe.scheduler = oldsched
        return latents
            
    
class UnetWrapper(torch.nn.Module):
    """ Necessary to wrap Unet because gradient checkpoint doesn't allow kwargs so need to make used args a tuple. """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        
    def forward(self, latent_model_input,
                    t,
                    encoder_hidden_states,
                    timestep_cond,
                    cross_attention_kwargs,
                    added_cond_kwargs,
                    return_dict,):
        ret = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict,)
        return ret
    
    
def latent_to_pil(latents, pipe) -> PIL.Image:
    """
    Simple utils for converting a latent to an image

    @param latents: torch.Tensor
    @param pipe: DiffusionPipelineProvider

    @return: PIL.Image
    """
    with torch.no_grad():
        image = pipe.vae.decode(latents.detach() / pipe.vae.config.scaling_factor, return_dict=False, generator=None)[0]
    # do_denormalize = [True] * image.shape[0]
    # ret = self.pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
    ret = [to_pil_image((img * 0.5 + 0.5).clamp(0, 1)) for img in image]
    return ret
    
    
def pixel_to_latent(image, pipe):
    """
    Simple utils for converting an image to a latent

    @param image: PIL.Image or torch.Tensor
    @param pipe: DiffusionPipelineProvider
    """
    if not isinstance(image, torch.Tensor):
        image = to_tensor(image).to(pipe.device)[None]
        image = 2. * image - 1.
    
    posterior = pipe.vae.encode(image).latent_dist
    z0 = posterior.mean
    z0 = z0.to(pipe.device)
    z0 = z0 * pipe.vae.config.scaling_factor  # * 0.18215 for SD 15, 21, Mitsua...
    
    return z0
    

def invert_image(pipe=None, scheduler=None,
                 image=None,
                 image_pt=None,
                 num_inference_steps=50,
                 guidance_scale=1,
                 resolution=512
                 ) -> torch.Tensor:
    """
    Simple util for inverting out of context of our pipe providers

    @param pipe: DiffusionPipelineProvider
    @param image: PIL.Image or torch.Tensor
    @param image_pt: torch.Tensor
    @param num_inference_steps: int
    @param guidance_scale: float
    @param resolution: int

    @return: torch.Tensor
    """
    # need to get z0 first
    if image_pt is None:
        image_pt = to_tensor(image).to(pipe.device)[None]
        image_pt = 2. * image_pt - 1.
    else:
        assert image is None
        
    posterior = pipe.vae.encode(image_pt).latent_dist
    z0 = posterior.mean
    z0 = z0.to(pipe.device)
    z0 = z0 * pipe.vae.config.scaling_factor  # * 0.18215 for SD 15, 21, Mitsua...

    # set to inverse scheduler
    orig_sched = pipe.scheduler
    pipe.scheduler = scheduler
    
    # invert z0 to zT
    zT_retrieved = pipe(latents=z0,
                        num_inference_steps=num_inference_steps,
                        prompt=[""] * z0.shape[0],
                        negative_prompt=[""],
                        guidance_scale=guidance_scale,
                        width=resolution,
                        height=resolution,
                        output_type='latent',
                        return_dict=False,
                        )[0]
    pipe.scheduler = orig_sched
    return zT_retrieved


def validate(
        out_dir: str,
        image_to_verify_PIL: PIL.Image,
        original_PIL: PIL.Image,
        wm_provider: WmProvider,
        pipe_provider_target,
        num_inference_steps_target,
        step,
        generated_PIL = None,
        message_bits_str_initial = None,
        do_psnr = True,
        do_ssim = True,
        do_msssim = True,
        do_lpips = True,
        lpips_loss_fn=None,
        device="cuda" if torch.cuda.is_available() else "cpu"
        ):
    """
    Perform a validation with the target model provider and watermark provider.
    Works only for Batch Size 1.

    @param results_dir: str, path to the directory where to save the results
    @param image_to_verify_PIL: PIL.Image, the image to verify
    @param original_PIL: PIL.Image, the original image
    @param wm_provider: WmProviders, the watermark provider
    @param pipe_provider_target: DiffusionPipelineProvider, the target model provider
    @param num_inference_steps_target: int, the number of inference steps for the target model
    @param step: int, the step of the attack
    @param generated_PIL: PIL.Image, the generated image
    @param message_bits_str_initial: str, the initial message bits
    @param do_psnr: bool, whether to calculate the psnr
    @param do_ssim: bool, whether to calculate the ssim
    @param do_msssim: bool, whether to calculate the msssim
    @param do_lpips: bool, whether to calculate the lpips
    @param lpips_loss_fn: callable, the lpips loss function, if None, will use the default one

    @return: dict, the results of the validation
    """

    # ------------------------- Validate -------------------------

    with torch.no_grad():
        # retrieve zT
        zT_retrieved = pipe_provider_target.invert_images(image_to_verify_PIL, num_inference_steps=num_inference_steps_target)["zT_torch"]

    # watermark test
    accuracy_results = wm_provider.get_accuracies(zT_retrieved)

    # Gaussian Shading bit accuracy
    bit_accuracy = accuracy_results["bit_accuracies"][0] if "bit_accuracies" in accuracy_results else 0.0
    message_bits_str_recovered = accuracy_results["message_bits_str_list"][0] if "message_bits_str_list" in accuracy_results else None

    # Tree-Ring p-value
    p_value = accuracy_results["p_values"][0] if "p_values" in accuracy_results else 0.0

    # psnr
    if do_psnr:
        psnr = psnr_PIL(image_to_verify_PIL, original_PIL)
    else:
        psnr = -1

    # ssim
    if do_ssim:
        ssim = ssim_PIL(image_to_verify_PIL, original_PIL)[0]
    else:
        ssim = -1

    # msssim
    if do_msssim:
        msssim = msssim_PIL(image_to_verify_PIL, original_PIL)
    else:
        msssim = -1

    # lpips
    if do_lpips:
        lpips = lpips_PIL(image_to_verify_PIL, original_PIL, loss_fn=lpips_loss_fn, device=device)
    else:
        lpips = -1

    # ------------------------ Save results ---------------------

    # preare a dir to save the results
    os.makedirs(out_dir, exist_ok=True)

    # save the original image
    original_PIL.save(os.path.join(out_dir, "original.png"))
    # save the generated image. This is empty in the case of removal attack because there, the original image is the generated image
    if generated_PIL is not None:
        generated_PIL.save(os.path.join(out_dir, "generated.png"))

    # save the image for the current step
    image_to_verify_PIL.save(os.path.join(out_dir, f"attack_instance_step={step}.png"))

    # save the diff between attack instance and original
    if do_psnr:
        diff = to_pil_image(torch.abs(to_tensor(image_to_verify_PIL) - to_tensor(original_PIL)) )
        diff.save(os.path.join(out_dir, f"diff_step={step}.png"))

    # collect metrics to a row for a csv with results
    return {
        # meta
        "step": step,
        "psnr": psnr,
        "ssim": ssim,
        "msssim": msssim,
        "lpips": lpips,
        # initial
        "message_bits_str_initial": message_bits_str_initial,
        # retrieved
        "bit_accuracy": bit_accuracy,
        "p_value": p_value,
        "message_bits_str_recovered": message_bits_str_recovered,
        }


