import torch
import typing
from diffusers import FluxPipeline

import PIL

from .pipe_provider import PipeProvider

from utils.image_utils import torch_to_PIL, torch_to_PIL

from typing import List, Optional, Union
import torch
from diffusers import FluxPipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import *

import numpy as np


DTYPE = torch.bfloat16  # FLUX needs bfloats. Not all GPUs support bfloats. If you get an error, try torch.float32
    
class FlowMatchEulerDiscreteSchedulerInverse(FlowMatchEulerDiscreteScheduler):
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        sigmas = torch.flip(sigmas, [0])
        timesteps = sigmas * self.config.num_train_timesteps

        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([torch.zeros(1, device=sigmas.device), sigmas])

        self._step_index = None
        self._begin_index = None


def latent_to_pil(latents, height, width, pipe):
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image


def pil_to_latent(image, height, width, pipe):
    image = pipe.image_processor.preprocess(image).to(dtype=torch.bfloat16,
                                                      device="cuda"
                                                      )
    latents = pipe.vae.encode(image, return_dict=False)[0].sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latents = pipe._pack_latents(latents,
                                 1,
                                 16,
                                 height // pipe.vae_scale_factor * 2,
                                 width // pipe.vae_scale_factor * 2)
    return latents


class FluxPipeProvider(PipeProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call_from_pretrained(self,
                             pretrained_model_name_or_path: str,
                             **kwargs) -> FluxPipeline:
        """
        Call Diffusers "from_pretrained"

        @param pretrained_model_name_or_path: str
        @param kwargs: dict

        @return: DiffusionPipeline
        """
        pipe_class = self.get_diffusers_pipe_class()
        pipe = pipe_class.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                          safety_checker=None,
                                          torch_dtype=self.get_dtype(),
                                          **kwargs
                                          )
        pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU memory

        return pipe


    # -------------------------------------------------------- GENERATE + INVERSION --------------------------------------------------------

    def generate(self,
                 prompts: typing.List[str],
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 latents: typing.Optional[torch.Tensor] = None,
                 return_latents: bool = True) -> dict:
        """
        Generate dict with

        @param prompt: str
        @param num_inference_steps: int
        @param guidance_scale: float
        @param latents: torch.Tensor with batch dim
        @param return_latents: bool

        @return dict
        """

        if not isinstance(prompts, list):
            prompts = [prompts]

        zT = latents 

        zT_shape = self.get_latent_shape(batch_size=1)
        zT_prepared = zT.view(zT_shape[0] * zT_shape[1] * zT_shape[2], zT_shape[3])

        z0_prepared = self.pipe(
            prompts,
            height=self.resolution,
            width=self.resolution,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="latent",
            latents=zT_prepared[None],
        ).images[0]
        
        with torch.no_grad():
            image = latent_to_pil(z0_prepared[None], height=self.resolution, width=self.resolution, pipe=self.pipe)[0]
    
        # collect results
        images_PIL = [image] # list of PIL image
        images_torch = self.PIL_to_torch([image], dtype=torch.float32)

        if return_latents:
            z0_torch = z0_prepared.view(zT_shape)
            z0_PIL = torch_to_PIL(torch.zeros((1, 4, self.resolution // 8, self.resolution // 8)))

            zT_torch = zT.to(self.device) if zT is not None else torch.zeros(z0_torch.shape).to(self.device)
            zT_PIL = torch_to_PIL(torch.zeros((1, 4, self.resolution // 8, self.resolution // 8)))
        
        return {
            # prompts
            "prompts": prompts,  # is list
            # images
            'images_torch': images_torch,
            'images_PIL': images_PIL,
            'images': images_PIL,
            # z0
            'z0_torch': z0_torch if return_latents else None,
            'z0_PIL': z0_PIL if return_latents else None,
            'z0': z0_PIL if return_latents else None,
            # zT
            'zT_torch': zT_torch if return_latents else None,
            'zT_PIL': zT_PIL if return_latents else None,
            'zT': zT_PIL if return_latents else None,}
    
    def invert_z0(self,
                  latents: torch.Tensor,
                  num_inference_steps: int = 20) -> torch.tensor:
        """
        Do DDIM inversion on given latents z0

        @param latents: torch tensor with batch dim
        @param num_inference_steps: int

        @return: zT torch tensor with batch dim on self.device

        LATENTS HAS SHAPE 1, 1024, 64
        """

        # invert
        self.pipe.scheduler.__class__ = FlowMatchEulerDiscreteSchedulerInverse
        zT_inv = self.pipe(
            "",
            latents=latents,
            height=self.resolution,
            width=self.resolution,
            guidance_scale=1,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            #generator=torch.Generator("cpu").manual_seed(0),
            output_type="latent",
        ).images[0]
        
        zT_shape = self.get_latent_shape(batch_size=1)
        zT_inv = zT_inv.view(zT_shape)
        zT_inv = zT_inv.to(dtype=torch.float32)
        
        # reset back to default scheduler
        self.pipe.scheduler.__class__ = FlowMatchEulerDiscreteScheduler

        #return out
        return zT_inv

    def invert_images(self,
                      images: typing.Union[PIL.Image.Image,
                                           typing.List[PIL.Image.Image],
                                           torch.Tensor] = None,
                      num_inference_steps: int = 50,
                      skip_inversion: bool = False,
                      latents: torch.Tensor = None):
        """
        Do DDIM inversion on given images
        
        Accepts PIL, list of PIL, or
        torch tensor with or without batch dim, in [0, 1]

        @param images: PIL, list of PIL, or torch tensor with or without batch dim, in [0, 1]
        @param num_inference_steps: int
        @param skip_inversion: bool
        @param latents: latents

        @return: dict
        """

        assert images is not None or latents is not None, "Either images or latents must be given"

        # cast to torch on device
        if isinstance(images, PIL.Image.Image) or isinstance(images, list):
            images = self.PIL_to_torch(images)

        # collect results
        z0_torch = pil_to_latent(images, self.resolution, self.resolution, self.pipe) if latents is None else latents
        z0_PIL = torch_to_PIL(torch.zeros((1, 4, self.resolution // 8, self.resolution // 8)))

        zT_torch = self.invert_z0(z0_torch, num_inference_steps=num_inference_steps) if not skip_inversion else z0_torch
        zT_PIL = torch_to_PIL(torch.zeros((1, 4, self.resolution // 8, self.resolution // 8)))
        
        return {
            'z0_torch': z0_torch,
            'z0_PIL': z0_PIL,
            'z0': z0_PIL,
            'zT_torch': zT_torch,
            'zT_PIL': zT_PIL,
            'zT': zT_PIL}

    # -------------------------------------------------------- UTILS --------------------------------------------------------

    def imgs_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        return self.pipe.vae.encode(images, return_dict=False)[0]

    def set_scheduler(self):
        pass

    def get_diffusers_pipe_class(cls):
        return FluxPipeline
    
    def get_latent_shape(self, batch_size=1):
        return (batch_size, 16, self.resolution // 8, self.resolution // 8)
    
    def get_random_latents(self, batch_size=1) -> torch.Tensor:
        return torch.randn(*self.get_latent_shape(batch_size=batch_size),
                           device=self.device,
                           dtype=self.get_dtype())
    
    def get_vae_params(self):
        return None
    
    def get_unet_params(self):
        return None
    
    def get_scheduler(self):
        return None

    def get_inverse_scheduler(self):
        return None
    
    def allow_device(self):
        """
        Allow manual device placement
        """
        return False
    
    def get_dtype(self):
        return DTYPE
