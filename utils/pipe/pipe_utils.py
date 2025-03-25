
import torch

from .schedulers.scheduling_ddim import DDIMScheduler
from .schedulers.scheduling_ddim_inverse import DDIMInverseScheduler

from diffusers import DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler

from .SD_provider import SDPipeProvider
from .SDXL_provider import SDXLPipeProvider
from .PixArt_provider import PixArtPipeProvider



# Map model id onto pipe. Add more if needed
PIPE_PROVIDERS = {
    'stabilityai/stable-diffusion-2-1-base': SDPipeProvider,
    'stabilityai/stable-diffusion-xl-base-1.0': SDXLPipeProvider,
    'PixArt-alpha/PixArt-Sigma-XL-2-512-MS': PixArtPipeProvider,
    # FLUX see below
    }


SCHEDULER_CLASSES = {
     "DDIM": (DDIMScheduler, DDIMInverseScheduler),
     "DPM": (DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler),
     "Euler": (None, None),  # special case for Flux
}


def get_pipe_provider(pretrained_model_name_or_path: str,
                      resolution: int,  # only square image
                      unet_id_or_checkpoint_dir: str = None,
                      lora_checkpoint_dir: str = None,
                      vae_id: str = None,
                      zero_unet: bool = False,
                      device: torch.device = torch.device("cuda"),
                      eager_loading: bool = False,
                      schedulers_name: str = "DDIM",
                      **kwargs):
        """
        Get correct pipe provider

        @param pretrained_model_name_or_path: str
        @param resolution: int
        @param unet_id_or_checkpoint_dir: str
        @param lora_checkpoint_dir: str
        @param vae_id: str
        @param zero_unet: bool
        @param device: torch.device
        @param eager_loading: bool
        @param schedulers_name: str
        @param kwargs: dict

        @return: PipeProvider
        """

        if "FLUX" in pretrained_model_name_or_path:
             from .Flux_provider import FluxPipeProvider
             PIPE_PROVIDERS['black-forest-labs/FLUX.1-dev'] = FluxPipeProvider
    
        # get correct pipe with all possible arguments. Args are then disseminated in the subclasses
        pipe_provider = PIPE_PROVIDERS[pretrained_model_name_or_path]
        return pipe_provider(pretrained_model_name_or_path=pretrained_model_name_or_path,
                             resolution=resolution,
                             unet_id_or_checkpoint_dir=unet_id_or_checkpoint_dir,
                             lora_checkpoint_dir=lora_checkpoint_dir,
                             vae_id=vae_id,
                             zero_unet=zero_unet,
                             device=device,
                             eager_loading=eager_loading,
                             scheduler_classes=SCHEDULER_CLASSES[schedulers_name],
                             **kwargs)
