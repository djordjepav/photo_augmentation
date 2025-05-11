from PIL import Image, ImageFilter


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
        
    Raises
    ------
    ValueError
        If the mask and content images have different dimensions
        
    Notes
    -----
    - Both input images are converted to RGBA if they aren't already
    - The mask should be a single-channel (grayscale) image
    - Larger kernel_size values create smoother transitions
    - Final output is converted back to RGB mode
    """

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

