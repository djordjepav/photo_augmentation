import base64
import io
import logging
from enum import Enum

import numpy as np
import torch
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from ..models.detection import PersonDetectionModel
from ..models.diffusion import PersonGeneratingModel, PhotoInpaintingModel
from ..models.segmentation import PersonSegmentationModel
from ..metrics.metrics import calculate_lpips, calculate_ssim
from ..utils import (
    blend_masks,
    create_person_mask,
    crop_and_upsample_mask,
    crop_to_divisible,
    plot_mask_contours,
    resize_if_large,
)
from . import app


class Prompts(Enum):
    INPAINT_PERSON = "perfect face symmetry, matching skin tone to original photo, \
        seamless integration with original lighting, natural skin pores, photorealistic, 8k detail, \
        identical white balance, matching contrast and saturation levels, seamless edge blending"
    INPAINT_PERSON_NEGATIVE = "deformed, blurry, bad anatomy, disfigured, poorly drawn face, \
        mutated hands, fused fingers, disconnected limbs, extra limbs, cloned face, \
        asymmetric eyes, unnatural skin tone, overexposed, underexposed, lowres, jpeg artifacts, \
        watermark, text, cartoon, 3d, doll, anime, monochrome, style mismatch"
    INPAINT_SCENE = ""
    INPAINT_SCENE_NEGATIVE = ""
    GENERATE_PERSON = "High-resolution portrait, photorealistic face, \
        detailed skin texture with pores, natural lighting, symmetrical features, \
        realistic eyes with catchlights, subtle skin imperfections, soft bokeh background, \
        professional studio photo, 8K detail"
    GENERATE_PERSON_NEGATIVE = "deformed, blurry, asymmetric, bad anatomy, disfigured, \
        cloned face, mutated hands, unnatural skin tone, plastic texture, cartoon, anime, \
        3D render, doll, text, watermark, lowres, extra limbs, two heads, extra heads"


# Initialize logger.
logger = logging.getLogger(__name__)

# Initialize models.
inpaintor = PhotoInpaintingModel()
generator = PersonGeneratingModel()
detector = PersonDetectionModel()
segmentor = PersonSegmentationModel()


# DONE
@app.post("/generate_person")
async def generate_person(
    prompt_image: UploadFile=File(None),
    prompt: str="",
    negative_prompt: str="",
):
    """
    Generate a photorealistic person image from text and/or image prompts.
    
    :param prompt_image: Reference image for IP-Adapter conditioning.
    :param prompt: Text description of the desired person.
    :param negative_prompt: Text description of elements to avoid.
    :returns: Generated person image in PNG format.
    """

    try:
        # Default image size.
        width, height = 512, 512
        
        print(prompt_image)
        
        # Prepare inputs.
        if prompt_image is not None:
            prompt_image = Image.open(io.BytesIO(await prompt_image.read()))
            prompt_image = resize_if_large(prompt_image)
            prompt_image = crop_to_divisible(prompt_image)

            width, height = prompt_image.size

        # Generate person.
        prompt += ", " + str(Prompts.GENERATE_PERSON)
        negative_prompt += ", " + str(Prompts.GENERATE_PERSON_NEGATIVE)

        person = generator.generate_person(
            prompt_image=prompt_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height
        )

        # Convert and return result.
        img_byte_arr = io.BytesIO()
        person.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.exception("An error occurred") 
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/swap_person")
async def swap_person(
    image: UploadFile=File(...),
    prompt_image: UploadFile=File(None),
    prompt_generation: str="",
    negative_prompt_generation: str="",
    prompt_inpainting: str="",
    negative_prompt_inpainting: str="",
    use_generated_person: bool=False,
    return_metrics: bool=False,
    person_index: int=0,
    debug_mode: bool=False,
):
    """
    Replace a person in an image with a generated or reference person.
    
    :param image: Base image containing person to replace.
    :param prompt_image: Reference image for person generation.
    :param prompt_generation: Text prompt for person generation.
    :param negative_prompt_generation: Negative prompt for person generation.
    :param prompt_inpainting: Text prompt for inpainting.
    :param negative_prompt_inpainting: Negative prompt for inpainting.
    :param use_generated_person: Whether to use generated person.
    :param return_metrics: Whether to return quality metrics.
    :param person_index: Index of person to replace.
    :param debug_mode: Whether to return debug visualization.
    :returns: Modified image or dict with image and metrics.
    """

    try:
        # Prepare inputs.
        image = Image.open(io.BytesIO(await image.read()))
        image = resize_if_large(image)
        image = crop_to_divisible(image)
        
        if prompt_image is not None:
            prompt_image = Image.open(io.BytesIO(await prompt_image.read()))
            prompt_image = resize_if_large(prompt_image)
            prompt_image = crop_to_divisible(prompt_image)
        
        # Detection and segmentation on people on photo.
        bboxes = detector.detect(image)        
        masks = segmentor.segment(image, bboxes)

        # Handle index out of range.
        # Reset to the last one.
        if person_index >= len(masks):
            person_index = len(masks) - 1
        mask = masks[person_index]
        
        # Calculate person size.
        y, x = np.where(mask > 0)
        
        height = np.max(y) - np.min(y)
        height = height - (height % 8)
        
        width = np.max(x) - np.min(x)
        width = width - (width % 8)
        
        # Generate new person based on reference one.    
        if use_generated_person:
            prompt_generation += ", " + str(Prompts.GENERATE_PERSON)
            negative_prompt_generation += ", " + str(Prompts.GENERATE_PERSON_NEGATIVE)
            
            person = generator.generate_person(
                prompt_image=prompt_image,
                prompt=prompt_generation, 
                negative_prompt=negative_prompt_generation,
                width=width,
                height=height
            )
            prompt_image = person
        else:
            prompt_inpainting += ", " + str(Prompts.GENERATE_PERSON)
            negative_prompt_inpainting += ", " + str(Prompts.GENERATE_PERSON_NEGATIVE)
                
        # Inpaint person on image.
        prompt_inpainting += ", " + str(Prompts.INPAINT_PERSON)
        negative_prompt_inpainting += ", " + str(Prompts.INPAINT_PERSON_NEGATIVE)

        inapainted_image = inpaintor.inpaint(
            image=image,
            mask=mask,
            prompt_image=prompt_image,
            prompt=prompt_inpainting,
            negative_prompt=negative_prompt_inpainting,
            strength=0.99,
            guidance_scale=7.5,
            ip_adapter_scale=0.9,
            blend_scale=5,
            inpaint_full_res_padding=64,
        )
        
        # TODO: Change to PIL methods.
        if debug_mode:
            # Temporary stitched solution.
            final_image = np.zeros(
                (inapainted_image.size[1], inapainted_image.size[0] * 2, 3), dtype=np.uint8
            )
            final_image[:, :inapainted_image.size[0], :] = np.asarray(inapainted_image)
            final_image[
                :prompt_image.size[1], 
                inapainted_image.size[0]:inapainted_image.size[0]+prompt_image.size[0], 
                :
            ] = np.asarray(prompt_image)
            final_image = Image.fromarray(final_image)
        else:
            final_image = inapainted_image

        # Prepare response.
        img_byte_arr = io.BytesIO()
        final_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
    
        # Return metrics if required.
        if return_metrics:
            # Convert to numpy for metrics.
            original_np = np.array(image)
            generated_np = np.array(inapainted_image)
            mask_np = np.array(mask)
            
            # Calculate metrics.
            ssim_score = float(calculate_ssim(original_np, generated_np, mask_np))
            lpips_score = float(calculate_lpips(
                torch.from_numpy(original_np), torch.from_numpy(generated_np)
            ))
            
            # Encode image.
            base64_img_byte_arr = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Rerutn response.
            return {
                "image": base64_img_byte_arr,
                "metrics": {
                    "ssim":ssim_score,
                    "lpips": lpips_score
                }
            }
        else:
            return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.exception("An error occurred") 
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/inject_person")
async def inject_person(
    image: UploadFile=File(...),
    prompt_image: UploadFile=File(None),
    prompt: str="",
    negative_prompt: str="",
    debug_mode: bool=False,
):
    """
    Inject a new person into an existing image.
    
    :param image: Base image to modify.
    :param prompt_image: Reference image for person generation.
    :param prompt: Text description of desired person.
    :param negative_prompt: Elements to avoid in output.
    :param debug_mode: Whether to return debug visualization.
    :returns: Modified image with injected person.
    """
    
    try:
        # Prepare inputs.
        image = Image.open(io.BytesIO(await image.read()))
        image = resize_if_large(image)
        image = crop_to_divisible(image)
        
        if prompt_image is not None:
            prompt_image = Image.open(io.BytesIO(await prompt_image.read()))
            prompt_image = resize_if_large(prompt_image)
            prompt_image = crop_to_divisible(prompt_image)

        # Perform detection and segmentation of foreground.
        bboxes = detector.detect(image)        
        masks = segmentor.segment(image, bboxes)

        # Generate mask where new person can be injected.
        mask, bbox = create_person_mask(masks)
        x, y, w, h = bbox
        
        # Generate new person based on reference photo.
        prompt = ", " + str(Prompts.GENERATE_PERSON)
        negative_prompt += ", " + str(Prompts.GENERATE_PERSON_NEGATIVE)
        
        person = generator.generate_person(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            prompt_image=prompt_image,
            width=h,
            height=w
        )
        
        # Get segmentation mask for generated person.
        person_bboxes = detector.detect(person)        
        person_mask = segmentor.segment(person, person_bboxes)[0]
        person_mask = Image.fromarray(person_mask).convert("L")
        
        # Resize generated person and its mask.
        person_mask, person = crop_and_upsample_mask(person_mask, person, w, h, True)

        # Change person mask's coordinate system.
        composite_mask = Image.new("L", mask.size, 0)
        composite_mask.paste(person_mask, (y, x))

        # Place new person behind the existing ones.
        composite_mask_np = np.asarray(composite_mask)
        mask_np = np.asarray(mask)        
        composite_mask_np = np.bitwise_and(composite_mask_np, mask_np)
        composite_mask = Image.fromarray(composite_mask_np).convert("L")
        
        composite = image.copy()
        composite.paste(person, (y, x))
        
        new_photo = blend_masks(image, composite, composite_mask, 1)
                                
        # Inpaint person on image.
        prompt = str(Prompts.INPAINT_PERSON) + ", " + prompt
        negative_prompt = str(Prompts.INPAINT_PERSON_NEGATIVE) + ", " + negative_prompt
            
        inapainted_image = inpaintor.inpaint(
            image=new_photo,
            mask=mask,
            prompt_image=person,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=0.2,
            guidance_scale=10,
            ip_adapter_scale=1,
            blend_scale=5,
        )
        
        # Send stitched image if in debug_mode.
        if debug_mode:
            final_image = np.zeros((new_photo.size[1], new_photo.size[0] * 2, 3), dtype=np.uint8)
            final_image[:, :new_photo.size[0], :] = np.asarray(new_photo)
            final_image[:, new_photo.size[0]:, :] = np.asarray(inapainted_image)
            final_image = Image.fromarray(final_image)
        else:
            final_image = inapainted_image

        # Convert and return result.
        img_byte_arr = io.BytesIO()
        final_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.exception("An error occurred") 
        raise HTTPException(status_code=500, detail="Internal server error.")
    

# DONE
@app.post("/inpaint_region")
async def inpaint_region(
    image: UploadFile=File(...),
    mask_x: int=0,
    mask_y: int=0,
    mask_width:int=128,
    mask_height:int=128,
    prompt: str="",
    negative_prompt: str="",
):
    """
    Inpaint a specified rectangular region of an image.
    
    :param image: Base image to modify.
    :param mask_x: X-coordinate of mask region.
    :param mask_y: Y-coordinate of mask region.
    :param mask_width: Width of mask region.
    :param mask_height: Height of mask region.
    :param prompt: Text description of desired content.
    :param negative_prompt: Elements to avoid in output.
    :returns: Image with inpainted region in PNG format.
    """

    try:
        # Prepare image.
        image = Image.open(io.BytesIO(await image.read()))
        image = resize_if_large(image)   
        image = crop_to_divisible(image)
        image = np.asarray(image).copy()

        # Mask region.
        mask = np.zeros(image.shape[:-1], dtype=np.uint8)

        mask[mask_x:mask_x+mask_width, mask_y:mask_y+mask_height] = 255        
        image[mask_x:mask_x+mask_width, mask_y:mask_y+mask_height, ...] = 0
        
        # Inpaint masked region.
        inpainted_image = inpaintor.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=1,
            guidance_scale=2
        )
        
        # Return result
        img_byte_arr = io.BytesIO()
        inpainted_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.exception("An error occurred") 
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/segment_person")
async def segment_person(
    image: UploadFile=File(...),
):
    """
    Detect and segment all persons in an image.
    
    :param image: Input image containing persons.
    :returns: Visualization of person segmentation masks.
    """

    try:
        # Load and adapt images.
        image = Image.open(io.BytesIO(await image.read()))
        image = resize_if_large(image)
        image = crop_to_divisible(image)
        
        # Perform detection and segmentation of foreground.
        bboxes = detector.detect(image)        
        masks = segmentor.segment(image, bboxes)
        
        # Visualize segmentation.
        result = plot_mask_contours(np.asarray(image), masks)
        result = Image.fromarray(result)
        
        # Convert and return result.
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.exception("An error occurred") 
        raise HTTPException(status_code=500, detail="Internal server error.")
