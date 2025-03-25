"""
Original by https://github.com/YuxinWenRick/tree-ring-watermark and heavily modified for debugging purposes and to get access to internals
Please give them credit and adhere to their license agreement.
"""

from tqdm import tqdm

import copy

import typing

import argparse

import numpy as np
import scipy

import torch

from .wm_provider import WmProvider

from utils.image_utils import torch_to_PIL
from utils import utils


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--w_seed', default=999999, type=int)
parser.add_argument('--w_channel', default=3, type=int)
parser.add_argument('--w_pattern', default='ring')
parser.add_argument('--w_mask_shape', default='circle')
parser.add_argument('--w_radius', default=10, type=int)
parser.add_argument('--w_measurement', default='l1_complex')
parser.add_argument('--w_injection', default='complex')
parser.add_argument('--w_pattern_const', default=0, type=float)


class TrProvider(WmProvider):
    """
    Original by https://github.com/YuxinWenRick/tree-ring-watermark and heavily modified for debugging purposes and to get access to internals
    """

    def __init__(self,
                 w_seed: int = None,
                 w_channel: int = 3,
                 w_pattern: str = 'ring',
                 w_mask_shape: str = 'circle',
                 w_radius: int = 10,
                 w_measurement: str = 'l1_complex',
                 w_injection: str = 'complex',
                 w_pattern_const: float = 0,
                 **kwargs):
        """
        @param w_seed: int, seed for watermarking
        @param w_channel: int, channel to watermark
        @param w_pattern: str, pattern to watermark
        @param w_mask_shape: str, shape of the mask
        @param w_radius: int, radius of the watermark
        @param w_measurement: str, measurement to use
        @param w_injection: str, injection to use
        @param w_pattern_const: float, constant for the pattern
        """
        super().__init__(**kwargs)

        # This ensures, every latent ever create has the same WM pattern
        # This makes sense when we simulate only one service provider using one kind of WM pattern
        if w_seed is not None:
            utils.set_random_seed(w_seed)

        self.w_seed = w_seed
        self.w_channel = w_channel
        self.w_pattern = w_pattern
        self.w_mask_shape = w_mask_shape
        self.w_radius = w_radius
        self.w_measurement = w_measurement
        self.w_injection = w_injection
        self.w_pattern_const = w_pattern_const

        # these are of shape self.latent_shape
        # so they can deal with batches just fine
        self.gt_patch = self.__get_watermarking_pattern()
        self.watermarking_mask = self.__get_watermarking_mask()


    def get_wm_type(self) -> str:
        return "TR"


    def fft_get_wchannel(self, images: torch.Tensor) -> torch.tensor:
        """
        Do fft, return only the important channel, and return it as numpy array

        @param img: torch tensor with batch dim or without batch dim.

        @return img: np.ndarray
        """
        if len(images.shape) < 4:
            images = images.unsqueeze(0)
        images = torch.fft.fftshift(torch.fft.fft2(images), dim=(-1, -2))[:, self.w_channel].real
    
        return images


    def __circle_mask(self, size=64, r=10, x_offset=0, y_offset=0) -> np.ndarray:
        """
        reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3

        @param size: int, size of the mask
        @param r: int, radius of the circle
        @param x_offset: int, x offset
        @param y_offset: int, y offset

        @return: np.ndarray
        """
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
    
        return ((x - x0)**2 + (y-y0)**2)<= r**2


    def __get_watermarking_mask(self) -> torch.tensor:
        """
        Original by https://github.com/YuxinWenRick/tree-ring-watermark

        @return: torch.tensor on self.device
        """
        watermarking_mask = torch.zeros(self.latent_shape, dtype=torch.bool).to(self.device)
    
        if self.w_mask_shape == 'circle':
            np_mask = self.__circle_mask(self.latent_shape[-1], r=self.w_radius)
            torch_mask = torch.tensor(np_mask).to(self.device)
    
            if self.w_channel == -1:
                # all channels
                watermarking_mask[:, :] = torch_mask
            else:
                watermarking_mask[:, self.w_channel] = torch_mask
        elif self.w_mask_shape == 'square':
            anchor_p = self.latent_shape[-1] // 2
            if self.w_channel == -1:
                # all channels
                watermarking_mask[:, :, anchor_p-self.w_radius:anchor_p+self.w_radius, anchor_p-self.w_radius:anchor_p+self.w_radius] = True
            else:
                watermarking_mask[:, self.w_channel, anchor_p-self.w_radius:anchor_p+self.w_radius, anchor_p-self.w_radius:anchor_p+self.w_radius] = True
        elif self.w_mask_shape == 'no':
            pass
        else:
            raise NotImplementedError(f'w_mask_shape: {self.w_mask_shape}')
    
        return watermarking_mask


    def __get_watermarking_pattern(self) -> torch.tensor:
        """
        Get the watermarking pattern

        @return: torch.tensor on self.device
        """
        
        gt_init = torch.randn(*self.latent_shape, device=self.device)
    
        if 'seed_ring' in self.w_pattern:  # unused
            gt_patch = gt_init
    
            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = self.__circle_mask(gt_init.shape[-1], r=i)
                tmp_mask = torch.tensor(tmp_mask).to(self.device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        elif 'seed_zeros' in self.w_pattern:  # unused
            gt_patch = gt_init * 0
        elif 'seed_rand' in self.w_pattern:  # unused
            gt_patch = gt_init
        elif 'rand' in self.w_pattern:  # unused
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch[:] = gt_patch[0]
        elif 'zeros' in self.w_pattern:  # unused
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        elif 'const' in self.w_pattern:  # unused
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
            gt_patch += self.w_pattern_const
        elif 'ring' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = self.__circle_mask(gt_init.shape[-1], r=i)
                tmp_mask = torch.tensor(tmp_mask).to(self.device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    
        return gt_patch
    

    def __inject_watermark(self, latents_clean: torch.tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject watermark into the latents

        @param latents_clean: torch.Tensor, shape: self.latent_shape,

        @return: tuple
            init_latents_w, torch.Tensor, is dtype=torch.float32, shape: self.latent_shape, on self.device
            init_latents_w_fft_pristine, torch.Tensor, is dtype=torch.float32, shape: self.latent_shape, on self.device
        """
    
        # This necessary if we choose float pipeline in the beginning - Their random latents will be floar 16 too.
        # This will be problematic once we do FFT: FFT of float 16 is complex 32. We need it to be complex 64 though
        # because gt_patch is complex 64.
        latents_w = copy.deepcopy(latents_clean).float()
    
        latents_w_fft = torch.fft.fftshift(torch.fft.fft2(latents_w), dim=(-1, -2))
    
        if self.w_injection == 'complex':
            latents_w_fft[self.watermarking_mask] = self.gt_patch[self.watermarking_mask].clone()
            latents_w = torch.fft.ifft2(torch.fft.ifftshift(latents_w_fft, dim=(-1, -2)))
            latents_w = latents_w.real  # Here is a big mistake in the original code. They shouldn't just drop the complex part
        elif self.w_injection == 'seed':
            latents_w[self.watermarking_mask] = self.t_patch[self.watermarking_mask].clone()
        else:
            NotImplementedError(f'w_injection: {self.w_injection}')

        pristine_latents_w_fft = latents_w_fft  # we keep the fft of before the fft-inverse drops its complex part
        return latents_w, pristine_latents_w_fft


    def get_wm_latents(self,
                       latents_clean: torch.Tensor = None,
                       seed: int = None) -> typing.Dict[str, any]:
        """
        Get the latents for the watermarking scheme

        @param latents_clean: torch.Tensor, shape: self.latent_shape,
        @param seed: int, seed for watermarking

        @return: dict
        """
        if seed is not None:
            utils.set_random_seed(seed)

        # latent can be given or we just generate ad hoc
        if latents_clean is None:
            latents_clean = torch.randn(self.latent_shape)
        latents_clean = latents_clean.clone().to(self.device, self.dtype)
        
        # inject watermark
        # we also get the fft of the pristine fft, with the mistake of dropping complex part of the fft-inverse made by original authors
        latents_w, pristine_latents_w_fft = self.__inject_watermark(latents_clean)
        # drop complex part
        pristine_latents_w_fft = pristine_latents_w_fft.real

        # get PIL images too
        # clean
        latents_clean_torch = latents_clean.to(self.device)
        latents_clean_PIL = torch_to_PIL(latents_clean_torch)
        # clean fft
        latents_clean_fft_torch = torch.fft.fftshift(torch.fft.fft2(latents_clean.to(torch.float32)), dim=(-1, -2)).real.to(self.device)
        latents_clean_fft_PIL = torch_to_PIL(latents_clean_fft_torch)
        # clean fft wchannel
        latents_clean_fft_wchannel_torch = latents_clean_fft_torch[:, self.w_channel: self.w_channel + 1]
        latents_clean_fft_wchannel_PIL = torch_to_PIL(latents_clean_fft_wchannel_torch)


        # watermarked
        latents_w_torch = latents_w.to(self.device)
        latents_w_PIL = torch_to_PIL(latents_w_torch)
        # watermarked fft
        latents_w_fft_torch = torch.fft.fftshift(torch.fft.fft2(latents_w_torch), dim=(-1, -2)).real.to(self.device)
        latents_w_fft_PIL = torch_to_PIL(latents_w_fft_torch)
        # watermarked fft wchannel
        latents_w_fft_wchannel_torch = latents_w_fft_torch[:, self.w_channel: self.w_channel + 1].to(self.device)
        latents_w_fft_wchannel_PIL = torch_to_PIL(latents_w_fft_wchannel_torch)
        # watermarked fft pristine
        pristine_latents_w_fft_torch = pristine_latents_w_fft.to(self.device)
        pristine_latents_w_fft_PIL = torch_to_PIL(pristine_latents_w_fft_torch)
        # watermarked fft wchannel pristine
        pristine_latents_w_fft_wchannel_torch = pristine_latents_w_fft[:, self.w_channel: self.w_channel + 1].to(self.device)
        pristine_latents_w_fft_wchannel_PIL = torch_to_PIL(pristine_latents_w_fft_wchannel_torch)

        return {
            # clean
            "zT_clean_torch": latents_clean_torch,
            "zT_clean_PIL": latents_clean_PIL,
            "zT_clean": latents_clean_PIL,
            # clean fft
            "zT_clean_fft_torch": latents_clean_fft_torch,
            "zT_clean_fft_PIL": latents_clean_fft_PIL,
            "zT_clean_fft": latents_clean_fft_PIL,
            # clean fft wchannel
            "zT_clean_fft_wchannel_torch": latents_clean_fft_wchannel_torch,
            "zT_clean_fft_wchannel_PIL": latents_clean_fft_wchannel_PIL,
            "zT_clean_fft_wchannel": latents_clean_fft_wchannel_PIL,

            # watermarked
            "zT_torch": latents_w_torch,
            "zT_PIL": latents_w_PIL,
            "zT": latents_w_PIL,
            # watermarked fft
            "zT_fft_torch": latents_w_fft_torch,
            "zT_fft_PIL": latents_w_fft_PIL,
            "zT_fft": latents_w_fft_PIL,
            # watermarked fft wchannel
            "zT_fft_wchannel_torch": latents_w_fft_wchannel_torch,
            "zT_fft_wchannel_PIL": latents_w_fft_wchannel_PIL,
            "zT_fft_wchannel": latents_w_fft_wchannel_PIL,
            # for the watermartked images, we also have a original fft which occured during watermark injection
            # it contains information which is later dropped when in fft-inverse, the complex part is dropped
            # !pristine! watermarked fft 
            "pristine_zT_fft_torch": pristine_latents_w_fft_torch,
            "pristine_zT_fft_PIL": pristine_latents_w_fft_PIL,
            "pristine_zT_fft": pristine_latents_w_fft_PIL,
            # !pristine! watermarked fft wchannel
            "pristine_zT_fft_wchannel_torch": pristine_latents_w_fft_wchannel_torch,
            "pristine_zT_fft_wchannel_PIL": pristine_latents_w_fft_wchannel_PIL,
            "pristine_zT_fft_wchannel": pristine_latents_w_fft_wchannel_PIL,
            }
    

    def __get_p_value(self,
                      latents: torch.tensor) -> typing.List[float]:
        """
        Low means -> Watermark is present

        Will calulate the fft of latetns and search for WM pattern

        @param latent: torch.tensor, shape: self.latent_shape, on device

        @return: dict
        """
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
        
        watermarking_mask = self.watermarking_mask[0]
        gt_patch = self.gt_patch[0]

        # Just iterate samples in batch, is not that expensive anyways
        p_values = []
        for latent_fft in latents_fft:

            # get the watermarking mask
            latent_fft = latent_fft[watermarking_mask].flatten()
            target_patch = gt_patch[watermarking_mask].flatten()
            target_patch = torch.concatenate([target_patch.real, target_patch.imag])
            latent_fft = torch.concatenate([latent_fft.real, latent_fft.imag])
    
            # p test
            simga = latent_fft.std()
            lambd = (target_patch ** 2 / simga ** 2).sum().item()
            x = (((latent_fft - target_patch) / simga) ** 2).sum().item()
            p = scipy.stats.ncx2.cdf(x=x, df=len(target_patch), nc=lambd)
            p_values.append(p)

        latents_fft_torch = latents_fft.real
        latents_fft_PIL = torch_to_PIL(latents_fft_torch)

        latents_fft_wchannel_torch = latents_fft_torch[:, self.w_channel: self.w_channel + 1]
        latents_fft_wchannel_PIL = torch_to_PIL(latents_fft_wchannel_torch)
    
        return {"p_values": p_values,
                "zT_fft_torch": latents_fft_torch,
                "zT_fft_PIL": latents_fft_PIL,
                "zT_fft": latents_fft_PIL,
                "zT_fft_wchannel_torch": latents_fft_wchannel_torch,
                "zT_fft_wchannel_PIL": latents_fft_wchannel_PIL,
                "zT_fft_wchannel": latents_fft_wchannel_PIL}
    

    def get_accuracies(self,
                       latents: typing.Union[torch.Tensor, np.array],) -> typing.Dict[str, any]:
        """
        Get the accuracy of the watermarking scheme

        @param latents: torch.Tensor or np.array, shape: self.latent_shape,

        @return: dict
        """
        results = self.__get_p_value(latents)

        p_values = results["p_values"]

        accuracies = [1 - p for p in p_values]

        return {
            "accuracies": accuracies,
            "p_values": p_values,
            "zT_fft_torch": results["zT_fft_wchannel_PIL"],
            "zT_fft_PIL": results["zT_fft_PIL"],
            "zT_fft": results["zT_fft"],
            "zT_fft_wchannel_torch": results["zT_fft_wchannel_PIL"],
            "zT_fft_wchannel_PIL": results["zT_fft_wchannel_PIL"],
            "zT_fft_wchannel": results["zT_fft_wchannel"]
        }