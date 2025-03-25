from diffusers import StableDiffusionPipeline

from .pipe_provider import PipeProvider


class SDPipeProvider(PipeProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_diffusers_pipe_class(cls):
        return StableDiffusionPipeline
    

