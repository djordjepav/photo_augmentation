from typing import Union, Optional
from time import time

import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionInpaintPipeline
from transformers import CLIPVisionModelWithProjection

from .utils import blend_masks


class PersonGeneratingModel:
    """A model for generating photorealistic person images using Stable Diffusion with IP-Adapter.
    
    This class handles the generation of person images either from text prompts alone or
    with additional image guidance through IP-Adapter.
    """
    
    def __init__(
        self,
        model_version: str="./v1-5-pruned.safetensors",
        ip_adapter_version: str="h94/IP-Adapter",
        ip_adapter_weights: str="ip-adapter_sd15.bin",
        device: Optional[str]=None
    ):
        """Initialize the person generation model.
        
        Parameters
        ----------
        model_version : str, optional
            Path or identifier for the Stable Diffusion model weights, 
            by default "./v1-5-pruned.safetensors"
        ip_adapter_version : str, optional
            Repository ID for IP-Adapter weights, 
            by default "h94/IP-Adapter"
        ip_adapter_weights : str, optional
            Specific weight file name for IP-Adapter, 
            by default "ip-adapter_sd15.bin"
        device : Optional[str], optional
            Device to run the model on ('cuda' or 'cpu'), 
            by default None (auto-detected)
        """

        # Initialize pipeline parameters.
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

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
        prompt: str="",
        negative_prompt: str="",
        image_prompt: Union[Image.Image, np.ndarray, None]=None,
        width: int=512,
        height: int=512,
        num_inference_steps: int=30,
        guidance_scale: float=7.5,
        ip_adapter_scale: float=0.7,
        seed: Optional[int]=None
    ) -> Image.Image:
        """Generate a photorealistic person image from text and/or image prompts.
        
        Parameters
        ----------
        prompt : str, optional
            Text description of the desired person, by default ""
        negative_prompt : str, optional
            Text description of elements to avoid, by default ""
        image_prompt : Union[Image.Image, np.ndarray, None], optional
            Reference image for IP-Adapter conditioning, by default None
        width : int, optional
            Output image width in pixels, by default 512
        height : int, optional
            Output image height in pixels, by default 512
        num_inference_steps : int, optional
            Number of denoising steps, by default 30
        guidance_scale : float, optional
            Text guidance strength (higher = more prompt influence), by default 7.5
        ip_adapter_scale : float, optional
            IP-Adapter conditioning strength (0-1), by default 0.7
        seed : Optional[int], optional
            Random seed for reproducibility, by default None
        
        Returns
        -------
        Image.Image
            Generated person image
        """

        # Random generator.
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.manual_seed(int(time()))
        
        # Inference.
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=image_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ip_adapter_scale=ip_adapter_scale,
            generator=generator,
        ).images[0]
        
        # TODO: Handle empty image prompts.
        
        return image


class PhotoInpaintingModel:
    """A model for inpainting persons into photos using Stable Diffusion Inpainting with IP-Adapter.
    
    This class handles the integration of persons into existing images while maintaining
    consistency with the original lighting, style, and composition.
    """

    def __init__(
        self,
        model_version: str="runwayml/stable-diffusion-inpainting", 
        ip_adapter_version: str="h94/IP-Adapter",
        ip_adapter_weights: str="ip-adapter-plus_sd15.safetensors",
        device: Optional[str]=None
    ):
        """Initialize the inpainting model.
        
        Parameters
        ----------
        model_version : str, optional
            Stable Diffusion inpainting model identifier, 
            by default "runwayml/stable-diffusion-inpainting"
        ip_adapter_version : str, optional
            Repository ID for IP-Adapter weights, 
            by default "h94/IP-Adapter"
        ip_adapter_weights : str, optional
            Specific weight file name for IP-Adapter, 
            by default "ip-adapter-plus_sd15.safetensors"
        device : Optional[str], optional
            Device to run the model on ('cuda' or 'cpu'), 
            by default None (auto-detected)
        """

        # Initialize model parameters.
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Initialize base pipeline.
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_version, torch_dtype=torch_dtype
        ).to(device)

        # Initialize image encoder.
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            ip_adapter_version, subfolder="models/image_encoder", torch_dtype=torch_dtype
        ).to(device)
        
        # Initialize IP Adapter.
        self.pipe.load_ip_adapter(
            ip_adapter_version,
            subfolder="models",
            weight_name=ip_adapter_weights,
            image_encoder=image_encoder,
        )


    def inpaint_person(
        self,
        image: Union[Image.Image, np.ndarray],
        person: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray, None]=None,
        prompt: str="",
        negative_prompt: str=""
    ) -> Image.Image:
        """Inpaint a person into an existing image while preserving context.
        
        Parameters
        ----------
        image : Union[Image.Image, np.ndarray]
            Base image to modify
        person : Union[Image.Image, np.ndarray]
            Person image to inpaint
        mask : Union[Image.Image, np.ndarray, None], optional
            Mask defining the inpainting region, by default None
        prompt : str, optional
            Text guidance for the inpainting, by default ""
        negative_prompt : str, optional
            Elements to avoid in the output, by default ""
        
        Returns
        -------
        Image.Image
            The inpainted image with blended edges
        """

        # Prepare inputs.
        out_image = image.copy()
        if isinstance(out_image, np.ndarray):
            out_image = Image.fromarray(out_image)
            
        if isinstance(person, np.ndarray):
            person = Image.fromarray(person)

        width, height = out_image.size
        
        if mask is None:
            mask = np.zeros((width, height))
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask).convert("L")

        # Inference.
        result = self.pipe(
            prompt=prompt + ", in a group photo, naturally blended",
            negative_prompt=negative_prompt,
            ip_adapter_image=person,
            image=out_image,
            mask_image=mask,
            height=height,
            width=width,
            strength=0.7,
            guidance_scale=7.5,
            num_inference_steps=30,
            inpaint_full_res=True,
            inpaint_full_res_padding=32
        ).images[0]

        # Blend masks.
        output = blend_masks(image, result, mask, 10)

        return output


    def inpaint_region(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
        prompt: str="",
        negative_prompt: str=""
    ) -> Image.Image:
        """Inpaint a masked region of an image with content matching the prompt.
        
        Parameters
        ----------
        image : Union[Image.Image, np.ndarray]
            Base image to modify
        mask : Union[Image.Image, np.ndarray]
            Mask defining the region to inpaint
        prompt : str, optional
            Text description of desired content, by default ""
        negative_prompt : str, optional
            Elements to avoid in the output, by default ""
        
        Returns
        -------
        Image.Image
            The inpainted image with blended edges
        """

        # Prepare input images.
        out_image = image.copy()
        if isinstance(out_image, np.ndarray):
            out_image = Image.fromarray(out_image)
            
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask).convert("L")

        # Inference.
        width, height = out_image.size

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=out_image,
            ip_adapter_image=out_image,
            ip_adapter_scale=0,
            mask_image=mask,
            height=height,
            width=width,
            strength=0.7,
            guidance_scale=7.5,
            num_inference_steps=30,
            inpaint_full_res=True,
            inpaint_full_res_padding=32
        ).images[0]
        
        # Blend masked region.
        output = blend_masks(image, result, mask, 10)

        return output


    def inject_person(
        self,
        image: Union[Image.Image, np.ndarray], 
        person: Union[Image.Image, np.ndarray], 
        mask: Union[Image.Image, np.ndarray, None]=None, 
        prompt: str="",
        negative_prompt: str=""
    ) -> Image.Image:
        """Inject a person into an image while matching lighting and style.
        
        Parameters
        ----------
        image : Union[Image.Image, np.ndarray]
            Target background image
        person : Union[Image.Image, np.ndarray]
            Person image to inject
        mask : Union[Image.Image, np.ndarray, None], optional
            Region where person should be placed, by default None
        prompt : str, optional
            Text description guiding the integration, by default ""
        negative_prompt : str, optional
            Elements to avoid in the output, by default ""
        
        Returns
        -------
        Image.Image
            The composited image with natural blending
        """

        # Prepare input images.
        out_image = image.copy()
        if isinstance(out_image, np.ndarray):
            out_image = Image.fromarray(out_image)
            
        if isinstance(person, np.ndarray):
            person = Image.fromarray(person)

        width, height = out_image.size
        
        if mask is None:
            mask = np.zeros((width, height))
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask).convert("L")

        # Inference.
        result = self.pipe(
            prompt=prompt + "a person naturally blending with the group, matching lighting and style",
            negative_prompt=negative_prompt,
            ip_adapter_image=person,
            image=out_image,
            mask_image=mask,
            height=height,
            width=width,
            strength=0.7,
            guidance_scale=7.5,
            num_inference_steps=30,
            inpaint_full_res=True,
            inpaint_full_res_padding=32
        ).images[0]

        # Blend masks.
        output = blend_masks(image, result, mask, 10)

        return output
