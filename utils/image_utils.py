import typing

import os

import math

from PIL import Image, ImageFilter

import torch

from torchvision import transforms

import numpy as np

import uuid

from PIL import Image
import numpy as np

from skimage.metrics import structural_similarity as ssim

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure

import lpips


def distort_images(images: typing.Union[Image.Image, typing.List[Image.Image]],
                   r_degree: float = None,
                   jpeg_ratio: int = None,
                   #jpeg_ratio_GS: int = None,
                   crop_scale_TR: float = None,
                   random_crop_ratio: float = None,
                   random_drop_ratio: float = None,
                   gaussian_blur_r: int = None,
                   gaussian_std: float = None,
                   gaussian_std_fixed: float = None,
                   median_blur_k: int = None,
                   sp_prob_GS: float = None,
                   sp_prob_fixed: float = None,
                   brightness_factor: float = None,
                   **kwargs
                   ) -> typing.Union[Image.Image, typing.List[Image.Image]]:
    """
    Distort image or list of images. Used for showing the impact of common transformations.
    Includes multiple versions of the same transformation becsaue some were incorrect, custom implementation of well known transformation in the Tree-Ring or Gaussian Shading repos.
    We fixed all the broken ones.

    @param img: PIL image or list of PIL images
    @param r_degree: float
    @param jpeg_ratio: int
    # @param jpeg_ratio_GS: int
    @param crop_scale_TR: float
    @param random_crop_ratio: float
    @param random_drop_ratio: float
    @param gaussian_blur_r: int
    @param gaussian_std: float
    @param gaussian_std_fixed: float
    @param median_blur_k: int
    @param sp_prob_GS: float
    @param sp_prob_fixed: float
    @param brightness_factor: float

    @return: PIL image or list of PIL images depending on what came in
    """
    if isinstance(images, Image.Image):
        was_wrapped = False
        images = [images]
    elif isinstance(images, list):
        was_wrapped = True
    else:
        raise ValueError("Input must be PIL image or list of PIL images")

    distorted_images = []
    for img in images:

        # from TR repo
        if r_degree is not None:
            img = transforms.RandomRotation((r_degree, r_degree))(img)
    
        # from TR repo, fixed by author. There was a possible race condition in the TR repo
        if jpeg_ratio is not None:
            file = f"OUT/{uuid.uuid4()}.jpg"
            img.save(file, quality=jpeg_ratio)
            img = Image.open(file)
            os.remove(file)
    
        # from TR repo
        # "Crop-Scale" in our paper
        if crop_scale_TR is not None:
            img = transforms.RandomResizedCrop(img.size,
                                               scale=(crop_scale_TR if crop_scale_TR is not None else 1,
                                                      crop_scale_TR if crop_scale_TR is not None else 1),
                                               ratio=(1, 1))(img)
            
        # from GS repo
        # "Random Crop" in our paper
        if random_crop_ratio is not None:
            # does some black bars which is unrealistic
            #set_random_seed(seed)
            width, height, c = np.array(img).shape
            img = np.array(img)
            new_width = int(width * random_crop_ratio)
            new_height = int(height * random_crop_ratio)
            start_x = np.random.randint(0, width - new_width + 1)
            start_y = np.random.randint(0, height - new_height + 1)
            end_x = start_x + new_width
            end_y = start_y + new_height
            padded_image = np.zeros_like(img)
            padded_image[start_y:end_y, start_x:end_x] = img[start_y:end_y, start_x:end_x]
            img = Image.fromarray(padded_image)
            
        # from GS repo
        # "Random Drop" in our paper
        if random_drop_ratio is not None:
            #set_random_seed(seed)
            width, height, c = np.array(img).shape
            img = np.array(img)
            new_width = int(width * random_drop_ratio)
            new_height = int(height * random_drop_ratio)
            start_x = np.random.randint(0, width - new_width + 1)
            start_y = np.random.randint(0, height - new_height + 1)
            padded_image = np.zeros_like(img[start_y:start_y + new_height, start_x:start_x + new_width])
            img[start_y:start_y + new_height, start_x:start_x + new_width] = padded_image
            img = Image.fromarray(img)

        # from GS & TR repos
        # "Gaussian Blur" in our paper
        if gaussian_blur_r is not None:
            img = img.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))

        # from GS repo
        # "Median Blur" in our paper
        if median_blur_k is not None:
            img = img.filter(ImageFilter.MedianFilter(median_blur_k))
    
        # from GS & TR repo. Is broken in both repos
        # not used in our paper
        if gaussian_std is not None:
            # old code does some weird clipping and extreme values
            img_shape = np.array(img).shape
            g_noise = np.random.normal(0, gaussian_std, img_shape) * 255
            g_noise = g_noise.astype(np.uint8)
            img = Image.fromarray(np.clip(np.array(img) + g_noise, 0, 255))
            
        # fixed by author, was wrongly implemented
        # "Gaussian STD" in our paper
        if gaussian_std_fixed is not None:
            img_tensor = transforms.ToTensor()(img)  # Converts to [0, 1] range, shape: [C, H, W]
            g_noise = torch.randn_like(img_tensor) * gaussian_std_fixed
            noisy_img_tensor = torch.clamp(img_tensor + g_noise, 0, 1)
            img = transforms.ToPILImage()(noisy_img_tensor)

        # from GS repo. Is broken
        # not used in our paper
        if sp_prob_GS is not None:
            # old code does x1.5 times the noise it is supposed to do
            c,h,w = np.array(img).shape
            prob_zero = sp_prob_GS / 2
            prob_one = 1 - prob_zero
            rdn = np.random.rand(c,h,w)
            img = np.where(rdn > prob_one, np.zeros_like(img), img)
            img = np.where(rdn < prob_zero, np.ones_like(img)*255, img)
            img = Image.fromarray(img)

        # fixed by author, was wrongly implemented in GS repo
        # "Salt & Pepper" in our paper
        if sp_prob_fixed is not None:
            # This may cause trouble with some numpy version so we only import it here
            import imgaug.augmenters as iaa

            img_np = np.array(img)
            augmenter = iaa.SaltAndPepper(sp_prob_fixed)
            img_noisy = augmenter(image=img_np)
            img = Image.fromarray(img_noisy)

        # from GS & TR
        # "Brightness" in our paper
        # Note that this actually does a random brightness change
        if brightness_factor is not None:
            img = transforms.ColorJitter(brightness=brightness_factor)(img)

        distorted_images.append(img)

    return distorted_images if was_wrapped else distorted_images[0]


def resize_tensor(img: torch.Tensor, resolution: int, seed: int = None) -> torch.Tensor:
    "Use this for resizing torch tensors - handles 3D (C,H,W) or 4D (B,C,H,W)"
    
    # Handle 3D tensor (C, H, W) by adding batch dimension
    if img.dim() == 3:
        img = img.unsqueeze(0)  # Add batch dimension
        remove_batch = True
    elif img.dim() == 4:
        remove_batch = False
    else:
        raise ValueError("Input tensor must be 3D (C,H,W) or 4D (B,C,H,W)")
    
    # Get the current image size
    _, _, height, width = img.shape
    
    # If image is smaller than desired resolution, resize it first
    if width < resolution or height < resolution:
        # Scale up maintaining aspect ratio
        scale = max(resolution / width, resolution / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = torch.nn.functional.interpolate(img, size=(new_height, new_width), mode='bilinear', align_corners=False)
    
    # Apply random crop to get the desired resolution
    if seed is not None:
        torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.RandomCrop(resolution),
    ])
    # Apply transform to each image in the batch
    result = torch.stack([transform(img_single) for img_single in img])
    
    # Remove batch dimension if input was 3D
    if remove_batch:
        result = result.squeeze(0)
    
    return result
    

def resize_PIL(img: Image.Image, resolution: int, seed: int = None) -> Image.Image:
    "Use this for resizing PIL images"

    # Get the current image size
    width, height = img.size
    
    # If image is smaller than desired resolution, resize it first
    if width < resolution or height < resolution:
        # Scale up maintaining aspect ratio
        scale = max(resolution / width, resolution / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Apply random crop to get the desired resolution
    if seed is not None:
        torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.RandomCrop(resolution),
    ])
    return transform(img)


def load_pil(filename: str, dir_name: str = "cache"):
    """
    load a PIL image from a file
    """

    # Full path for loading the image
    full_path = os.path.join(dir_name, filename)
    
    # Load and return the image
    return Image.open(full_path)


def save_pil(image: Image.Image, filename: str, dir_name: str = "cache"):
    """
    Save a PIL image to a file
    """
    
    # Create directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Full path for saving the image
    full_path = os.path.join(dir_name, filename)
    
    image.save(full_path)
    
    # Load the image to verify
    loaded_image = load_pil(filename)
    
    # Verify by comparing the two images
    assert list(image.getdata()) == list(loaded_image.getdata()), "Image not saved correctly"


def scale_tensor_to_range(tensor: torch.Tensor,
                          min_val: float = 0.0,
                          max_val: float = 1.0) -> torch.Tensor:
    """
    Scale a tensor to a given range.

    @param tensor: tensor to scale
    @param min_val: minimum value of the range
    @param max_val: maximum value of the range

    @return tensor: scaled tensor
    """
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = tensor * (max_val - min_val) + min_val

    return tensor


def torch_to_PIL(images: typing.Union[torch.Tensor, np.ndarray],
                 scale_to_pixel_vals: bool = True) -> typing.List[Image.Image]:
    """
    Images will be scaled to [0, 1]. All images in batch will be considered for determining this range.
    
    @param images: torch tensor with or without batch dim in [0, 1]. Also allows numpies, will be casted immediately
    @param scale_to_pixel_vals: bool, if True, will scale to [0, 255] and cast to uint8

    @return images: list of PIL images
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

      # SD3
    if images.shape[1] == 16:
            # Option 1: Average across the 16 latent channels to reduce to 3 channels
            images = images.mean(dim=1, keepdim=True)  # Mean over the 16 channels, keep dimension
            images = images.expand(-1, 3, -1, -1)  # Expand to 3 channels

    if scale_to_pixel_vals:
        images = scale_tensor_to_range(images, 0, 1)
        images = images * 255
        images = images.to(torch.uint8)
    images = images.detach().cpu()

    # Ensure the input is 4D (batch, channel, height, width) or 3D (channel, height, width)
    if images.dim() not in [3, 4]:
        raise ValueError("Input tensor must be 3D or 4D")

    # Prepare to convert each image in the batch
    if images.dim() == 4:
        # Batch of multi-channel images
        # colormap chosen automatically here. 1 channel -> grayscale, 3 channels -> RGB, 4 channels -> RGBA, more channels -> not enforced
        return [transforms.functional.to_pil_image(img) for img in images]
    else:
        # Batch of greyscale images
        # colormap chosen automatically here. 1 channel -> grayscale, 3 channels -> RGB, 4 channels -> RGBA, more channels -> not enforced
        return [transforms.functional.to_pil_image(img) for img in images]


def PIL_to_torch(images: typing.Union[Image.Image,
                                      typing.List[Image.Image]],
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Accepts PIL, list of PIL,
    One or more images to torch tensor with batch dim

    @param images: PIL, list of PIL
    @param dtype: dtype
    @param device: device

    @return latents: latents with batch dim on cpu
    """
    transform = transforms.ToTensor()

    if isinstance(images, Image.Image):
        images = transform(images)
    elif isinstance(images, list):
        images = torch.stack([transform(i) for i in images])

    return images.to(dtype=dtype, device=device)


def l2_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Computes the L2 (Euclidean) distance between two tensors.

    Args:
        tensor1 (torch.Tensor): The first tensor.
        tensor2 (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: The L2 distance between the two tensors.
    """
    return torch.norm(tensor1 - tensor2, p=2).item()


def ssim_PIL(img1: Image.Image, img2: Image.Image):
    """
    Calculate the SSIM (Structural Similarity Index) between two PIL images.
    
    Args:
        img1 (Image): First input image.
        img2 (Image): Second input image.
    
    Returns:
        float: The SSIM index value.
        ndarray: The full SSIM image.
    """

    # assert PIL
    if not isinstance(img1, Image.Image) or not isinstance(img2, Image.Image):
        raise ValueError("Both inputs must be PIL Image objects.")
    
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')

    # Ensure images are converted to numpy arrays and are in grayscale
    img1_np = np.array(img1.convert('L'))  # Convert image to grayscale
    img2_np = np.array(img2.convert('L'))  # Convert image to grayscale
    
    # Calculate SSIM
    ssim_index, ssim_map = ssim(img1_np, img2_np, full=True)
    
    return ssim_index, ssim_map


# method for MS-SSIM
def msssim_PIL(img1: Image.Image, img2: Image.Image):
    """
    Calculate the Multi-Scale Structural Similarity Index (MS-SSIM) between two PIL images using a reference implementation.

    Args:
        img1 (Image.Image): First input image.
        img2 (Image.Image): Second input image.
        weights (list, optional): Weights for each scale. If None, default weights are used by the reference implementation.

    Returns:
        float: The MS-SSIM index value.
    """

    # assert PIL
    if not isinstance(img1, Image.Image) or not isinstance(img2, Image.Image):
        raise ValueError("Both inputs must be PIL Image objects.")
    
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')

    # Convert images to torch tensors, normalized to [0,1], shape [1, C, H, W]
    arr1 = torch.tensor(np.array(img1)).permute(2, 0, 1).float() / 255.0
    arr2 = torch.tensor(np.array(img2)).permute(2, 0, 1).float() / 255.0
    arr1 = arr1.unsqueeze(0)
    arr2 = arr2.unsqueeze(0)

    msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    msssim_value = msssim_metric(arr1, arr2).item()
    return msssim_value


# add lpips using some reference implementation
def lpips_PIL(img1: Image.Image, img2: Image.Image, loss_fn, device) -> float:
    """
    Calculate the LPIPS (Learned Perceptual Image Patch Similarity) between two PIL images using the reference implementation.

    @param img1: The first PIL image.
    @param img2: The second PIL image.
    @param loss_fn: The LPIPS loss function, e.g., lpips.LPIPS(net='alex').
    @param device: The device to run the computation on, e.g., "cpu" or "cuda".

    @return: The LPIPS value between the two images.
    """

    if loss_fn is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = lpips.LPIPS(net='alex')
    loss_fn = loss_fn.to(device=device)

    # assert PIL
    if not isinstance(img1, Image.Image) or not isinstance(img2, Image.Image):
        raise ValueError("Both inputs must be PIL Image objects.")

    # Convert images to torch tensors, normalized to [-1, 1], shape [1, 3, H, W]
    arr1 = torch.tensor(np.array(img1.convert('RGB'))).to(device=device).permute(2, 0, 1).float() / 255.0
    arr2 = torch.tensor(np.array(img2.convert('RGB'))).to(device=device).permute(2, 0, 1).float() / 255.0
    arr1 = arr1 * 2 - 1
    arr2 = arr2 * 2 - 1
    arr1 = arr1.unsqueeze(0)
    arr2 = arr2.unsqueeze(0)

    lpips_value = loss_fn(arr1, arr2).detach().cpu().item()
    return lpips_value


def psnr(tensor1: torch.Tensor, tensor2: torch.Tensor, max_pixel_value: float = 1.0) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) using TorchMetrics.

    Args:
        tensor1 (torch.Tensor): First image tensor (BxCxHxW or CxHxW).
        tensor2 (torch.Tensor): Second image tensor.
        max_pixel_value (float): Max pixel value (e.g., 1.0 for normalized, 255 for uint8).

    Returns:
        float: PSNR value.
    """
    psnr_metric = PeakSignalNoiseRatio(data_range=max_pixel_value)
        
    return psnr_metric(tensor1, tensor2).item()


def psnr_PIL(img1: Image, img2: Image) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two PIL images.

    @param img1: The first PIL image.
    @param img2: The second PIL image.
    
    @return: The PSNR value between the two images.
    """

    # assert PIL
    if not isinstance(img1, Image.Image) or not isinstance(img2, Image.Image):
        raise ValueError("Both inputs must be PIL Image objects.")

    arr1 = torch.tensor(np.array(img1)).permute(2, 0, 1).float() / 255.0
    arr2 = torch.tensor(np.array(img2)).permute(2, 0, 1).float() / 255.0
    
    # Add batch dimension
    arr1 = arr1.unsqueeze(0)
    arr2 = arr2.unsqueeze(0)
    
    # Compute PSNR with TorchMetrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    psnr_value = psnr_metric(arr1, arr2).item()

    return psnr_value
