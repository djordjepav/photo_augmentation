import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance


def calculate_ssim(original_img: np.ndarray, generated_img: np.ndarray, mask: np.ndarray):
    """Calculate SSIM between masked regions of two images.
    
    :param original_img: Original image array.
    :param generated_img: Generated image array.
    :param mask: Binary mask defining region to compare.
    :returns: SSIM score between the masked regions.
    """

    y, x = np.where(mask > 0)
    min_y, max_y = np.min(y), np.max(y)
    min_x, max_x = np.min(x), np.max(x)
    
    original_crop = original_img[min_y:max_y, min_x:max_x]
    generated_crop = generated_img[min_y:max_y, min_x:max_x]
    
    return ssim(original_crop, generated_crop, multichannel=True, win_size=3)


def calculate_fid(original_img: torch.Tensor, generated_img: torch.Tensor):
    """Calculate FID score between two images.
    
    :param original_img: Original image tensor.
    :param generated_img: Generated image tensor.
    :returns: FID score between the images.
    """

    original_img = original_img.permute(2, 0, 1).unsqueeze(0)
    generated_img = generated_img.permute(2, 0, 1).unsqueeze(0)
    
    fid = FrechetInceptionDistance(feature=2048)
    fid.update(original_img, real=True)
    fid.update(generated_img, real=False)
    return fid.compute()


def calculate_lpips(original_img: torch.Tensor, generated_img: torch.Tensor):
    """Calculate LPIPS (perceptual similarity) between two images.
    
    :param original_img: Original image tensor.
    :param generated_img: Generated image tensor.
    :returns: LPIPS perceptual similarity score.
    """

    original_img = original_img.permute(2, 0, 1).unsqueeze(0).float() / 128 - 1
    generated_img = generated_img.permute(2, 0, 1).unsqueeze(0).float() / 128 - 1
    
    loss_fn = lpips.LPIPS(net='alex')
    loss = loss_fn(original_img, generated_img).item()
        
    return loss
