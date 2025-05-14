from time import time
from typing import Optional, Union

import numpy as np
import torch
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline
)
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from ..utils import blend_masks


class PersonGeneratingModel:
    """
    A model for generating photorealistic person images using Stable Diffusion with IP-Adapter.
    """
    
    def __init__(
        self,
        model_version: str="./models/v1-5-pruned.safetensors",
        ip_adapter_version: str="h94/IP-Adapter",
        ip_adapter_weights: str="ip-adapter_sd15.bin",
        device: Optional[str]=None,
        seed: Optional[int]=None
    ):
        """
        Initialize the person generation model.
        
        :param model_version: Path or identifier for the Stable Diffusion model weights.
        :param ip_adapter_version: Repository ID for IP-Adapter weights.
        :param ip_adapter_weights: Specific weight file name for IP-Adapter.
        :param device: Device to run the model on ('cuda' or 'cpu').
        :param seed: Random seed for reproducibility.
        """

        # Initialize pipeline parameters.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.seed = seed

        # Initialize scheduler for SD pipeline.
        scheduler = EulerDiscreteScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="scheduler"
        )
        
        # Initialized SD pipeline with downloaded weights.
        self.pipe = StableDiffusionPipeline.from_single_file(
            model_version,
            local_files_only=False,
            scheduler=scheduler,
            torch_dtype=torch_dtype,
        ).to(self.device)
        
        self.pipe.load_ip_adapter(
            ip_adapter_version, subfolder="models", weight_name=ip_adapter_weights
        )


    def generate_person(
        self,
        prompt_image: Union[Image.Image, np.ndarray, None]=None,
        prompt: str="",
        negative_prompt: str="",
        strength: float=0.8,
        guidance_scale: float=7.5,
        ip_adapter_scale: float=0.7,
        width: int=512,
        height: int=512,
        num_inference_steps: int=30,
    ) -> Image.Image:
        """
        Generate a photorealistic person image from text and/or image prompts.
        
        :param prompt_image: Reference image for IP-Adapter conditioning.
        :param prompt: Text description of the desired person.
        :param negative_prompt: Text description of elements to avoid.
        :param strength: Strength of the generation effect.
        :param guidance_scale: Controls how much the prompt influences generation.
        :param ip_adapter_scale: Strength of IP-Adapter conditioning.
        :param width: Output image width in pixels.
        :param height: Output image height in pixels.
        :param num_inference_steps: Number of denoising steps.
        :returns: Generated person image.
        """

        # Random generator.
        generator = torch.Generator(device=self.device)
        if self.seed is not None:
            generator.manual_seed(self.seed)
        else:
            generator.manual_seed(int(time()))
        
        # Handle inference without image prompt.
        if prompt_image is None:
            prompt_image = Image.new("RGB", (height, width), 0)
            ip_adapter_scale = 0

        # Inference.
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=prompt_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            ip_adapter_scale=ip_adapter_scale,
            generator=generator,
        ).images[0]
        
        return image


class PhotoInpaintingModel:
    """
    A model for inpainting persons and backgrounds into photos using 
    Stable Diffusion Inpainting.
    """

    def __init__(
        self,
        model_version: str="runwayml/stable-diffusion-inpainting", 
        ip_adapter_version: str="h94/IP-Adapter",
        ip_adapter_weights: str="ip-adapter-plus_sd15.safetensors",
        device: Optional[str]=None,
        seed: Optional[int]=None
    ):
        """
        Initialize the inpainting model.
        
        :param model_version: Stable Diffusion inpainting model identifier.
        :param ip_adapter_version: Repository ID for IP-Adapter weights.
        :param ip_adapter_weights: Specific weight file name for IP-Adapter.
        :param device: Device to run the model on ('cuda' or 'cpu').
        :param seed: Random seed for reproducibility.
        """

        # Initialize model parameters.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.seed = seed
        
        # Initialize base pipeline.
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_version, torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        # Initialize image encoder.
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            ip_adapter_version, subfolder="models/image_encoder", torch_dtype=torch_dtype
        ).to(self.device)
        
        # Initialize IP Adapter.
        self.pipe.load_ip_adapter(
            ip_adapter_version,
            subfolder="models",
            weight_name=ip_adapter_weights,
            image_encoder=image_encoder,
        )


    def inpaint(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
        prompt_image: Union[Image.Image, np.ndarray, None]=None,
        prompt: str="",
        negative_prompt: str="",
        strength=0.8,
        guidance_scale=7.5,
        ip_adapter_scale: int=0,
        blend_scale: int=0,
        inpaint_full_res: bool=True,
        inpaint_full_res_padding: bool=32,
        num_inference_steps=30,
    ) -> Image.Image:
        """
        Inpaint a masked region of an image with content matching the prompt.
        
        :param image: Base image to modify.
        :param mask: Mask defining the region to inpaint.
        :param prompt_image: Reference image for IP-Adapter conditioning.
        :param prompt: Text description of desired content.
        :param negative_prompt: Elements to avoid in the output.
        :param strength: Strength of the inpainting effect.
        :param guidance_scale: Controls how much the prompt influences generation.
        :param ip_adapter_scale: Strength of IP-Adapter conditioning.
        :param blend_scale: Size of Gaussian blur kernel for edge blending.
        :param inpaint_full_res: Whether to process only the masked area at full resolution.
        :param inpaint_full_res_padding: Border around mask in pixels.
        :param num_inference_steps: Number of denoising steps.
        :returns: The inpainted image with blended edges.
        """

        # Random generator.
        generator = torch.Generator(device=self.device)
        if self.seed is not None:
            generator.manual_seed(self.seed)
        else:
            generator.manual_seed(int(time()))

        # Prepare input images.
        out_image = image.copy()
        if isinstance(out_image, np.ndarray):
            out_image = Image.fromarray(out_image)
            
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask).convert("L")
        
        # Handle inference without image prompt.
        if prompt_image is None:
            prompt_image = out_image.copy()
            ip_adapter_scale = 0

        # Inference.
        width, height = out_image.size

        result = self.pipe(
            image=out_image,
            mask_image=mask,
            height=height,
            width=width,
            strength=strength,
            guidance_scale=guidance_scale,
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=prompt_image,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=num_inference_steps,
            inpaint_full_res=inpaint_full_res,
            inpaint_full_res_padding=inpaint_full_res_padding,
            generator=generator
        ).images[0]
        
        # Blend masked region.
        result = blend_masks(image, result, mask, blend_scale)

        return result
