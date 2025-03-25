from .pipe_provider import PipeProvider

from diffusers import PixArtSigmaPipeline, DiffusionPipeline, Transformer2DModel, AutoencoderKL

#from diffusers import DDIMScheduler, DDIMInverseScheduler
from .schedulers.scheduling_ddim import DDIMScheduler
from .schedulers.scheduling_ddim_inverse import DDIMInverseScheduler


ROOTNAME = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"


class PixArtPipeProvider(PipeProvider):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call_from_pretrained(self,
                             pretrained_model_name_or_path: str,
                             **kwargs) -> DiffusionPipeline:
        """
        Call Diffusers "from_pretrained"

        @param pretrained_model_name_or_path: str
        @param kwargs: dict

        @return: DiffusionPipeline
        """

        pipe_class = self.get_diffusers_pipe_class()

        # load transformer seperately. PixArt needs this
        transformer = self.get_unet(pretrained_model_name_or_path)

        # pipe must be loaded from ROOTNAME
        pipe = pipe_class.from_pretrained(pretrained_model_name_or_path=ROOTNAME,
                                          transformer=transformer,
                                          scheduler=self.scheduler,
                                          safety_checker=None,
                                          torch_dtype=self.get_dtype(),
                                          **kwargs)
        
        del transformer

        return pipe
    
    def get_diffusers_pipe_class(self):
        return PixArtSigmaPipeline

    def get_scheduler(self):
        return self.scheduler_classes[0].from_pretrained(ROOTNAME, subfolder='scheduler', torch_dtype=self.get_dtype())

    def get_inverse_scheduler(self):
        return self.scheduler_classes[1].from_pretrained(ROOTNAME, subfolder='scheduler', torch_dtype=self.get_dtype())

    def get_unet(self, unet_id_or_checkpoint_dir: str):
        return Transformer2DModel.from_pretrained(unet_id_or_checkpoint_dir, subfolder='transformer', torch_dtype=self.get_dtype(), device=self.device)

    def get_vae(self, *args, **kwargs):
        return AutoencoderKL.from_pretrained(ROOTNAME, subfolder="vae", torch_dtype=self.get_dtype(), device=self.device)

    def get_vae_params(self):
        return None
    
    def get_unet_params(self):
        return None
