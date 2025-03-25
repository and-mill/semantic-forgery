import torch
from diffusers import StableDiffusionXLPipeline

from .pipe_provider import PipeProvider


class SDXLPipeProvider(PipeProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_diffusers_pipe_class(cls):
        return StableDiffusionXLPipeline
    
    def get_random_latents(self, batch_size=1) -> torch.Tensor:
        return torch.randn(*self.get_latent_shape(batch_size=batch_size),
                           device=self.device,
                           dtype=self.get_dtype())