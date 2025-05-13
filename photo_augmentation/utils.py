from typing import List

import numpy as np
from scipy import ndimage
from PIL import Image, ImageFilter, ImageDraw


def crop_to_divisible(image: Image.Image, divisor: int=8):
    """Crop an image to dimensions divisible by a specified number.
    
    Parameters
    ----------
    image : Image.Image
        Input PIL Image to be cropped
    divisor : int, optional
        The number that both width and height should be divisible by, 
        by default 8
        
    Returns
    -------
    Image.Image
        Cropped image with dimensions divisible by the specified divisor
        
    Notes
    -----
    - The cropping is done from the bottom-right corner
    - Maintains the original image's top-left coordinates
    - Useful for preparing images for neural networks that require specific dimensions
    """

    width, height = image.size
    new_width = width - (width % divisor)
    new_height = height - (height % divisor)
    return image.crop((0, 0, new_width, new_height))


def blend_masks(
    background: Image.Image, 
    content: Image.Image,
    mask: Image.Image,
    kernel_size: int=5
):
    """Blend two images together using a mask with feathered edges.
    
    Parameters
    ----------
    background : Image.Image
        Base image that will serve as the background
    content : Image.Image
        New content image to be blended onto the background
    mask : Image.Image
        Grayscale mask defining the blending region (white=show content, black=show background)
    kernel_size : int, optional
        Size of Gaussian blur kernel for edge feathering, by default 5
        Set to 0 for no feathering
        
    Returns
    -------
    Image.Image
        The blended RGB image

    Notes
    -----
    - Both input images are converted to RGBA if they aren't already
    - The mask should be a single-channel (grayscale) image
    - Larger kernel_size values create smoother transitions
    - Final output is converted back to RGB mode
    """
    # Prepare input images.
    
    if isinstance(background, np.ndarray):
        background = Image.fromarray(background)
        
    if isinstance(content, np.ndarray):
        content = Image.fromarray(content).convert("L")
        
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask).convert("L")

    # Convert images to RGBA if needed.
    if background.mode != 'RGBA':
        background = background.convert('RGBA')
    if content.mode != 'RGBA':
        content = content.convert('RGBA')
    
    # Feather mask edges.
    if kernel_size > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(kernel_size))
    
    # Create transparent version of new content.
    transparent_content = content.copy()
    
    print(transparent_content.size, mask.size)
    # Apply feathered mask to alpha channel.
    transparent_content.putalpha(mask)
    
    # Composite images.
    result = Image.alpha_composite(background, transparent_content)

    # Convert back to RGB if needed.
    return result.convert('RGB')


def resize_if_large(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Resize image if any dimension exceeds max_size while maintaining aspect ratio.
    
    Parameters
    ----------
        image: Input PIL Image
        max_size: Maximum allowed dimension (default: 1024)
        
    Returns
    -------
        Resized PIL Image (or original if already small enough)
    """
    
    # Return image if the size does not exceed max size.
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    
    # Calculate new dimensions preserving aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.LANCZOS)


def crop_and_upsample_mask(
    mask: Image.Image,
    image: Image.Image,
    min_width: int = 1024, 
    min_height: int = 1024,
    upscale: bool = False,
) -> Image.Image:
    """
    Resizes an image ONLY if it's smaller than target dimensions.
    Preserves aspect ratio and prevents upscaling by default.
    
    Args:
        image: Input PIL Image
        min_width: Minimum width threshold (default: 1024)
        min_height: Minimum height threshold (default: 1024)
        upscale: If True, allows upscaling small images (default: False)
        resample: Resampling filter (default: LANCZOS for high quality)
    
    Returns:
        Resized PIL Image (or original if already large enough)
    """
    
    # Crop the mask so that it fits without margins.
    mask_np = np.asarray(mask)
    image_np = np.asarray(image)
    
    x, y = np.where(mask_np > 0)
    
    # height = np.max(y) - np.min(y)
    # height = height - (height % 8)
    
    # width = np.max(x) - np.min(x)
    # width = width - (width % 8)
    
    # print(x, x+width, y, y+height)
    
     
    mask_np = mask_np[np.min(x):np.max(x), np.min(y):np.max(y)]
    mask = Image.fromarray(mask_np).convert("L")
    
    image_np = image_np[np.min(x):np.max(x), np.min(y):np.max(y)]
    image = Image.fromarray(image_np)
    
    # Continue with cropped image.
    height, width = mask.size
    print(min_width, min_height, width, height)
    
    # Check if image meets minimum dimensions
    if (width >= min_width and height >= min_height) and not upscale:
        return mask, image
    
    # Calculate scaling factor
    width_ratio = min_width / width
    height_ratio = min_height / height
    scale = max(width_ratio, height_ratio) if upscale else min(width_ratio, height_ratio)
    
    # Apply scaling only if needed (when scale > 1 for upscale=False)
    if scale > 1 or upscale:
        new_width = int(width * scale)
        new_height = int(height * scale)
        mask = mask.resize((new_height, new_width), resample=Image.Resampling.NEAREST)
        image = image.resize((new_height, new_width), resample=Image.Resampling.LANCZOS)
        return mask, image
    
    return mask, image


def create_person_mask(masks: List[np.ndarray]) -> Image.Image:

    # Calculate person size.
    mask = masks[0]
    x, y = np.where(mask > 0)
    
    height = np.max(y) - np.min(y)
    height = height - (height % 8)
    
    width = np.max(x) - np.min(x)
    width = width - (width % 8)
    
    print("W, H = ", width, height)
    
    # Create background mask.
    bg_mask = np.zeros(masks[0].shape, dtype=np.uint8)
    for m in masks:
        bg_mask = np.bitwise_or(bg_mask, m)
    bg_mask = 255 - bg_mask

    bg_mask[0, :] = 0
    bg_mask[-1, :] = 0
    bg_mask[:, 0] = 0
    bg_mask[:, -1] = 0
    
    dist_mask = ndimage.distance_transform_edt(bg_mask)
    x, y = np.unravel_index(np.argmax(dist_mask), dist_mask.shape)
    
    x = max(0, x - (width // 2))
    y = max(0, y - (height // 2))
        
    print("X, Y = ", x, y)
    
    fg_mask = Image.new("L", dist_mask.shape, 0)  # Black mask = keep original
    draw = ImageDraw.Draw(fg_mask)
    draw.ellipse((x, y, x + width, y + height), fill=255)  # White ellipse = inpaint here
    # fg_mask = fg_mask.filter(ImageFilter.GaussianBlur(5))  # Soft edges for blending
    
    
    fg = np.asarray(fg_mask, dtype=np.uint8).T
    mask = np.bitwise_and(bg_mask, fg)

    return Image.fromarray(mask).convert("L"), (x, y, width, height)