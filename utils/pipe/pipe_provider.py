from abc import ABC, abstractmethod
import typing

import torch

#from diffusers import DDIMScheduler, DDIMInverseScheduler
from .schedulers.scheduling_ddim import DDIMScheduler
from .schedulers.scheduling_ddim_inverse import DDIMInverseScheduler

from diffusers import UNet2DConditionModel, AutoencoderKL, DiffusionPipeline, Transformer2DModel

import PIL

from utils import utils
from ..image_utils import torch_to_PIL, PIL_to_torch


DTYPE = torch.float32
NUM_LATENT_CHANNELS = 4


# ######################################################################################################################################################################################################
# ------------------------------------------------------------------------------------------------- Main -----------------------------------------------------------------------------------------------
# ######################################################################################################################################################################################################
class PipeProvider(ABC):
    """
    Helps in housekeeping attributes and methods which are needed by all kinds of different pipelines.
    """
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 resolution: int,
                 device: torch.device,
                 scheduler_classes: typing.Tuple[any, any] = (DDIMScheduler, DDIMInverseScheduler),
                 eager_loading: bool = False,
                 **kwargs):

        # model id or path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        # prepare schedulers
        # load correct scheduler
        self.scheduler_classes = scheduler_classes
        self.scheduler = self.get_scheduler()
        self.scheduler_inverse = self.get_inverse_scheduler()

        self.resolution = resolution
        self.device = device

        # eager loading of pipe
        self.kwargs = kwargs  # for later loading
        self.pipe = self.load_diffusers_pipe(self.pretrained_model_name_or_path, **self.kwargs) if eager_loading else None
            

    @abstractmethod
    def get_diffusers_pipe_class(self):
        pass

    def get_dtype(self):
        return DTYPE

    # -------------------------------------------------------- LOAD PIPE --------------------------------------------------------

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
        pipe = pipe_class.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                          scheduler=self.scheduler,  # set normal forward scheduler per default
                                          safety_checker=None,
                                          torch_dtype=self.get_dtype(),
                                          **kwargs)
        return pipe

    def load_diffusers_pipe(self,
                              pretrained_model_name_or_path: str,
                              unet_id_or_checkpoint_dir: str,
                              lora_checkpoint_dir: str,
                              vae_id: str,
                              zero_unet: bool,
                              disable_tqdm: bool = False,
                              **kwargs) -> DiffusionPipeline:
        """
        Load Diffusers pipe
        
        @param pretrained_model_name_or_path: str
        @param unet_id_or_checkpoint_dir: str
        @param lora_checkpoint_dir: str
        @param vae_id: str
        @param zero_unet: bool
        @param disable_tqdm: bool
        @param kwargs: dict

        @return: DiffusionPipeline
        """
        kwargs = {}

        # load unet if param given
        if unet_id_or_checkpoint_dir is not None:
            kwargs["unet"] = self.get_unet(unet_id_or_checkpoint_dir)

        # plug vae
        if vae_id is not None:
            kwargs["vae"] = self.get_vae(vae_id)
        
        # load our custom diffusers pipe subclass
        pipe = self.call_from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)
        pipe = pipe.to(self.device) if self.allow_device() else pipe

        # disable tqdm if necessary
        if disable_tqdm:
            pipe.set_progress_bar_config(disable=True)
        
        # load lora weights if param given
        if lora_checkpoint_dir is not None:
            pipe.load_lora_weights(lora_checkpoint_dir, weight_name="pytorch_lora_weights.safetensors")
    
        # zero out all unet params
        if zero_unet:
            for param in pipe.unet.parameters():
                param.data.zero_()
            for name, param in pipe.unet.named_parameters():
                print(f"{name}: {param.data.sum()}")
        
        return pipe
    

    # -------------------------------------------------------- VAE --------------------------------------------------------

    def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Accepts PIL, list of PIL, or
        torch tensor with or without batch dim, in [0, 1], in self.get_dtype()

        @param images: PIL, list of PIL, or torch tensor with or without batch dim, in [0, 1]
        @return: latents with batch dim on self.device
        """
        # to torch if necessary
        if isinstance(images, PIL.Image.Image) or isinstance(images, list):
            images = self.PIL_to_torch(images)

        images = images.to(self.device)

        self.__load_pipe()

        images = 2. * images - 1.
        posterior = self.pipe.vae.encode(images).latent_dist
        latents = posterior.mean * self.pipe.vae.config.scaling_factor  # * 0.18215 for SD 15, 21, Mitusa
        latents = latents.to(self.device)

        return latents

    def imgs_to_latents(self, images: typing.Union[PIL.Image.Image,
                                                   typing.List[PIL.Image.Image],
                                                   torch.Tensor]) -> torch.Tensor:
        """
        Accepts PIL, list of PIL, or
        torch tensor with or without batch dim, in [0, 1], in self.get_dtype()

        1. Transform to torch.Tensor if necessary
        2. Put image to self.device
        3. get latent z0 with self.pipe.vae

        @param images: PIL, list of PIL, or torch tensor with or without batch dim, in [0, 1]
        @return latents: latents with batch dim on self.device
        """

        # to torch if necessary
        if isinstance(images, PIL.Image.Image) or isinstance(images, list):
            images = self.PIL_to_torch(images).to(self.device)
        
        images = images.to(self.device)

        # unsqueeze
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        # push through vae
        latents = self.vae_encode(images).detach()

        return latents
    

    def latents_to_imgs(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Accepts latents with batch dim on self.device

        @param latents: latents with batch dim on self.device
        @return: images with batch dim on self.device
        """
        self.__load_pipe()

        # push through vae
        images = self.pipe.vae.decode(latents).sample.detach()

        return images


    # -------------------------------------------------------- GENERATE + INVERSION --------------------------------------------------------

    def generate(self,
                 prompts: typing.List[str],
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 latents: typing.Optional[torch.Tensor] = None,
                 num_images_per_prompt: int = 1,
                 return_latents: bool = True) -> dict:
        """
        Generate dict with

        @param prompt: str
        @param num_inference_steps: int
        @param guidance_scale: float
        @param latents: torch.Tensor with batch dim
        @param num_images_per_prompt: int
        @param return_latents: bool

        @return dict
        """
        self.__load_pipe()

        self.set_scheduler()

        # make sure latents are on device
        if latents is not None:
            latents = latents.to(self.device)

        # generate
        out = self.pipe(
             prompts,
             num_images_per_prompt=num_images_per_prompt,
             guidance_scale=guidance_scale,
             num_inference_steps=num_inference_steps,
             height=self.resolution,
             width=self.resolution,
             latents=latents,
             )
    
        # collect results
        images_PIL = out.images # list of PIL image
        images_torch = self.PIL_to_torch(out.images)

        if return_latents:
            z0_torch = self.imgs_to_latents(out.images)
            z0_PIL = torch_to_PIL(z0_torch)

            zT_torch = latents.to(self.device) if latents is not None else torch.zeros(z0_torch.shape).to(self.device)
            zT_PIL = torch_to_PIL(latents) if latents is not None else torch.zeros(z0_torch.shape).to(self.device)
        
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
                  num_inference_steps: int = 50) -> torch.tensor:
        """
        Do DDIM inversion on given latents z0

        @param latents: torch tensor with batch dim
        @param num_inference_steps: int

        @return: zT torch tensor with batch dim on self.device
        """
        self.__load_pipe()

        # set to inverse scheduler
        self.pipe.scheduler = self.scheduler_inverse

        # invert z0 to zT
        out = self.pipe(latents=latents,
                        num_inference_steps=num_inference_steps,
                        prompt=[""] * latents.shape[0],
                        negative_prompt="",
                        guidance_scale=1.,
                        width=self.resolution,
                        height=self.resolution,
                        output_type='latent',
                        return_dict=False,
                        )[0]
        
        # reset back to default scheduler
        self.pipe.scheduler = self.scheduler

        return out

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
        z0_torch = self.imgs_to_latents(images) if latents is None else latents
        z0_PIL = torch_to_PIL(z0_torch)

        zT_torch = self.invert_z0(z0_torch, num_inference_steps=num_inference_steps) if not skip_inversion else z0_torch
        zT_PIL = torch_to_PIL(zT_torch)
        
        return {
            'z0_torch': z0_torch,
            'z0_PIL': z0_PIL,
            'z0': z0_PIL,
            'zT_torch': zT_torch,
            'zT_PIL': zT_PIL,
            'zT': zT_PIL}
    

    # -------------------------------------------------------- UTILS --------------------------------------------------------
    
    def allow_device(self):
        """
        Allow manual device placement
        """
        return True

    def set_scheduler(self):
        self.pipe.scheduler = self.scheduler

    def get_scheduler(self):
        #return DDIMScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder='scheduler', torch_dtype=self.get_dtype())
        return self.scheduler_classes[0].from_pretrained(self.pretrained_model_name_or_path, subfolder='scheduler', torch_dtype=self.get_dtype())

    def get_inverse_scheduler(self):
        #return DDIMInverseScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder='scheduler', torch_dtype=self.get_dtype())
        return self.scheduler_classes[1].from_pretrained(self.pretrained_model_name_or_path, subfolder='scheduler', torch_dtype=self.get_dtype())

    def stash_pipe(self):
        """Drop pipe to free up memory"""
        if self.pipe is not None:
            self.pipe = self.pipe.to(torch.device("cpu")) if self.allow_device() else self.pipe

    def __load_pipe(self):
        """Load pipe and push to device"""
        if self.pipe is None:
            self.pipe = self.load_diffusers_pipe(self.pretrained_model_name_or_path, **self.kwargs)
        self.pipe = self.pipe.to(self.device) if self.allow_device() else self.pipe

    def set_lora(self, lora_checkpoint_dir: str):
        self.__load_pipe()
        self.pipe.load_lora_weights(lora_checkpoint_dir, weight_name="pytorch_lora_weights.safetensors")

    def get_unet(self, unet_id_or_checkpoint_dir: str):

        # for transoformer based models we need special care
        if "PixArt" in unet_id_or_checkpoint_dir:
            unet = Transformer2DModel.from_pretrained(unet_id_or_checkpoint_dir, subfolder='transformer', torch_dtype=self.get_dtype())
        else:
            unet = UNet2DConditionModel.from_pretrained(unet_id_or_checkpoint_dir, subfolder="unet", torch_dtype=self.get_dtype())
        unet = unet.to(self.device) if self.allow_device() else unet
        return unet

    def set_unet(self, unet_id_or_checkpoint_dir: str):
        self.__load_pipe()
        self.pipe.unet = self.get_unet(unet_id_or_checkpoint_dir)

    def get_vae(self, vae_id: str):
        vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae", torch_dtype=self.get_dtype())
        vae = vae.to(self.device) if self.allow_device() else vae
        return vae

    def set_vae(self, vae_id: str):
        self.__load_pipe()
        self.pipe.vae = self.get_vae(vae_id)

    def get_num_latent_channels(self):
        return NUM_LATENT_CHANNELS

    def get_latent_resolution(self):
        self.__load_pipe()
        return self.resolution // self.pipe.vae_scale_factor

    def get_image_shape(self, batch_size=1) -> typing.Tuple[int, int]:
        return (batch_size,
                3,
                self.resolution,
                self.resolution)

    def get_latent_shape(self, batch_size=1) -> typing.Tuple[int, int, int, int]:
        return (batch_size,
                self.get_num_latent_channels(),
                self.get_latent_resolution(),
                self.get_latent_resolution())

    def get_random_latents(self, batch_size=1) -> torch.Tensor:
        return torch.randn(*self.get_latent_shape(batch_size=batch_size),
                           device=self.device,
                           dtype=self.get_dtype())

    def PIL_to_torch(self, images: typing.Union[PIL.Image.Image, typing.List[PIL.Image.Image]], dtype=None) -> torch.Tensor:
        """
        Accepts PIL, list of PIL,
        One or more images to torch tensor with batch dim
        
        @param images: PIL, list of PIL
        @return latents: latents with batch dim on self.device
        """
        dtype = self.get_dtype() if dtype is None else dtype    
        return PIL_to_torch(images, dtype=dtype, device=self.device)

    def get_vae_params(self):
        """
        Get vae params as flattened torch tensor on cpu
        """
        self.__load_pipe()
        return {k: v.cpu() for k, v in self.pipe.vae.state_dict().items()}
    
    def get_unet_params(self):
        """
        Get unet params as flattened torch tensor on cpu
        """
        self.__load_pipe()
        return {k: v.cpu() for k, v in self.pipe.unet.state_dict().items()}
