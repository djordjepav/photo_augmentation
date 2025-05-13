import base64
from enum import Enum

from fastapi import UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, Response
import io
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import logging
import torch

from . import app
from ..models.diffusion import PersonGeneratingModel, PhotoInpaintingModel
from ..models.detection import PersonDetectionModel
from ..models.segmentation import PersonSegmentationModel
from ..utils import crop_to_divisible, resize_if_large, create_person_mask, blend_masks, \
    crop_and_upsample_mask
from ..metrics.metrics import calculate_fid, calculate_lpips, calculate_ssim


class Prompts(Enum):
    INPAINT_PERSON = "perfect face symmetry, matching skin tone to original photo, \
        seamless integration with original lighting, natural skin pores, photorealistic, 8k detail"
    INPAINT_PERSON_NEGATIVE = "deformed, blurry, bad anatomy, disfigured, poorly drawn face, \
        mutated hands, fused fingers, disconnected limbs, extra limbs, cloned face, \
        asymmetric eyes, unnatural skin tone, overexposed, underexposed, lowres, jpeg artifacts, \
        watermark, text, cartoon, 3d, doll, anime, monochrome"
    INPAINT_SCENE = ""
    INPAINT_SCENE_NEGATIVE = ""
    GENERATE_PERSON = "High-resolution portrait, photorealistic face, \
        detailed skin texture with pores, natural lighting, symmetrical features, \
        realistic eyes with catchlights, subtle skin imperfections, soft bokeh background, \
        professional studio photo, 8K detail"
    GENERATE_PERSON_NEGATIVE = "deformed, blurry, asymmetric, bad anatomy, disfigured, \
        cloned face, mutated hands, unnatural skin tone, plastic texture, cartoon, anime, \
        3D render, doll, text, watermark, lowres, extra limbs"


logger = logging.getLogger(__name__)

inpaintor = PhotoInpaintingModel()
generator = PersonGeneratingModel()
detector = PersonDetectionModel()
segmentor = PersonSegmentationModel()


@app.post("/generate_person")
async def generate_person(
    prompt: str = "",
    negative_prompt: str = "",
    reference_photo: UploadFile = File(None),
):
    try:        
        reference_photo = Image.open(io.BytesIO(await reference_photo.read()))
        reference_photo = resize_if_large(reference_photo)
        reference_photo = crop_to_divisible(reference_photo)
        
        width, height = reference_photo.size
        
        person = generator.generate_person(
            prompt=str(Prompts.GENERATE_PERSON) + ", " + prompt, 
            negative_prompt=str(Prompts.GENERATE_PERSON_NEGATIVE) + ", " + negative_prompt,
            image_prompt=reference_photo,
            width=width,
            height=height
        )

        img_byte_arr = io.BytesIO()
        person.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.exception("An error occurred") 
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/swap_person")
async def swap_person(
    photo: UploadFile = File(...),
    reference_photo: UploadFile = File(None),
    return_metrics: bool =False,
    prompt_generation: str = "",
    negative_prompt_generation: str = "",
    prompt_inpainting: str = "",
    negative_prompt_inpainting: str = "",
    person_index: int = 0,
):
    try:
        # Receive data.
        photo = Image.open(io.BytesIO(await photo.read()))
        photo = resize_if_large(photo)
        photo = crop_to_divisible(photo)
        
        reference_photo = Image.open(io.BytesIO(await reference_photo.read()))
        reference_photo = resize_if_large(reference_photo)
        reference_photo = crop_to_divisible(reference_photo)
        
        # Detection and segmentation on people on photo.
        bboxes = detector.detect(photo)        
        masks = segmentor.segment(photo, bboxes)        
        mask = masks[person_index]
        
        # Calculate person size.
        y, x = np.where(mask > 0)
        
        height = np.max(y) - np.min(y)
        height = height - (height % 8)
        
        width = np.max(x) - np.min(x)
        width = width - (width % 8)

        # Generate new person based on reference one.
        person = generator.generate_person(
            prompt=str(Prompts.GENERATE_PERSON) + ", " + prompt_generation, 
            negative_prompt=str(Prompts.GENERATE_PERSON_NEGATIVE) + ", " + negative_prompt_generation,
            image_prompt=reference_photo,
            width=width,
            height=height
        )
        
        # Inpaint generated person. 
        new_photo = inpaintor.inpaint_person(
            image=photo,
            person=person,
            mask=mask,
            prompt=str(Prompts.INPAINT_PERSON) + ", " + prompt_inpainting,
            negative_prompt=str(Prompts.INPAINT_PERSON_NEGATIVE) + ", " + negative_prompt_inpainting
        )

        # Prepare response.
        img_byte_arr = io.BytesIO()
        new_photo.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
    
        # Return metrics if required.
        if return_metrics:
            # Convert to numpy for metrics.
            original_np = np.array(photo)
            generated_np = np.array(new_photo)
            mask_np = np.array(mask)
            
            # Calculate metrics.
            ssim_score = float(calculate_ssim(original_np, generated_np, mask_np))
            # fid_score = calculate_fid(torch.from_numpy(original_np), torch.from_numpy(generated_np))
            lpips_score = float(calculate_lpips(
                torch.from_numpy(original_np), torch.from_numpy(generated_np)
            ))
            
            # Encode image.
            base64_img_byte_arr = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            return {
                "image": base64_img_byte_arr,
                "metrics": {
                    "ssim":ssim_score,
                    # "fid": fid_score.item(),
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
    photo: UploadFile = File(...),
    reference_photo: UploadFile = File(None),
    prompt: str = "",
    negative_prompt: str = "",
):
    try:
        # Load and adapt images.
        photo = Image.open(io.BytesIO(await photo.read()))
        photo = resize_if_large(photo)
        photo = crop_to_divisible(photo)

        reference_photo = Image.open(io.BytesIO(await reference_photo.read()))
        reference_photo = resize_if_large(reference_photo)
        reference_photo = crop_to_divisible(reference_photo)
        
        # Perform detection and segmentation of foreground.
        bboxes = detector.detect(photo)        
        masks = segmentor.segment(photo, bboxes)

        # Generate mask where new person can be injected.
        mask, bbox = create_person_mask(masks)
        x, y, w, h = bbox
        
        # Generate new person based on reference photo.
        person = generator.generate_person(
            prompt=str(Prompts.GENERATE_PERSON) + ", " + prompt, 
            negative_prompt=str(Prompts.GENERATE_PERSON_NEGATIVE) + ", " + negative_prompt,
            image_prompt=reference_photo,
            width=h,
            height=w
        )
        
        # Get segmentation mask for generated person.
        person_bboxes = detector.detect(person)        
        person_mask = segmentor.segment(person, person_bboxes)[0]
        person_mask = Image.fromarray(person_mask).convert("L")
        
        # Resize generated person and its mask.
        print("BEFORE", person_mask.size)
        person_mask, person = crop_and_upsample_mask(person_mask, person, w, h, True)
        print("AFTER", person_mask.size)        

        # Change person mask's coordinate system.
        composite_mask = Image.new("L", mask.size, 0)
        composite_mask.paste(person_mask, (y, x))

        # Place new person behind the existing ones.
        composite_mask_np = np.asarray(composite_mask)
        mask_np = np.asarray(mask)        
        composite_mask_np = np.bitwise_and(composite_mask_np, mask_np)
        composite_mask = Image.fromarray(composite_mask_np).convert("L")
        
        composite = photo.copy()
        composite.paste(person, (y, x))
        
        new_photo = blend_masks(photo, composite, composite_mask, 1)
        
        # Inpaint person on image.
        new_photo = inpaintor.inject_person(
            image=new_photo,
            person=new_photo,
            mask=mask,
            prompt=str(Prompts.INPAINT_PERSON) + ", " + prompt,
            negative_prompt=str(Prompts.INPAINT_PERSON_NEGATIVE) + ", " + negative_prompt
        )

        # Convert and return result.
        img_byte_arr = io.BytesIO()
        new_photo.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.exception("An error occurred") 
        raise HTTPException(status_code=500, detail="Internal server error.")
    

@app.post("/inpaint_region")
async def inpaint_region(
    file: UploadFile=File(...),
    x: int=0,
    y: int=0,
    width:int=128,
    height:int=128,
    prompt: str="",
    negative_prompt: str="",
):
    try:
        # Prepare image and masked region.
        image = Image.open(io.BytesIO(await file.read()))
        image = resize_if_large(image)   
        image = crop_to_divisible(image)
        image = np.asarray(image).copy()

        mask = np.zeros(image.shape[:-1], dtype=np.uint8)
        print(mask.shape)
        image[x:x+width, y:y+height, ...] = 0
        mask[x:x+width, y:y+height] = 255
        
        # image[..., 0] = np.where(mask==255, image[..., 0], 0)
        # image[..., 1] = np.where(mask==255, image[..., 1], 0)
        # image[..., 2] = np.where(mask==255, image[..., 2], 0)
        image = Image.fromarray(image)
        
        # Inpaint masked region.
        new_image = inpaintor.inpaint_region(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt
        )

        # Return result
        img_byte_arr = io.BytesIO()
        new_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.exception("An error occurred") 
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/segment_person")
async def segment_person(
    photo: UploadFile = File(...),
    reference_photo: UploadFile = File(None),
    prompt: str = "",
    negative_prompt: str = "",
):
    try:
        # Load and adapt images.
        photo = Image.open(io.BytesIO(await photo.read()))
        photo = resize_if_large(photo)
        photo = crop_to_divisible(photo)

        reference_photo = Image.open(io.BytesIO(await reference_photo.read()))
        reference_photo = resize_if_large(reference_photo)
        reference_photo = crop_to_divisible(reference_photo)
        
        # Create fg mask
        x, y = 20, 50  # Position where the person should be inserted
        width, height = 128, 256  # Approximate size of the new person
        
        fg_mask = Image.new("L", photo.size, 0)  # Black mask = keep original
        draw = ImageDraw.Draw(fg_mask)
        draw.ellipse((x, y, x + width, y + height), fill=255)  # White ellipse = inpaint here
        # fg_mask = fg_mask.filter(ImageFilter.GaussianBlur(5))  # Soft edges for blending
        
        # Perform detection and segmentation of foreground.
        bboxes = detector.detect(photo)        
        masks = segmentor.segment(photo, bboxes)
        
        # Calculate person size.
        mask = masks[0]
        y, x = np.where(mask > 0)
        
        height = np.max(y) - np.min(y)
        height = height - (height % 8)
        
        width = np.max(x) - np.min(x)
        width = width - (width % 8)
        
        print(width, height)
        
        # Merge masks and invert them.
        mask = np.zeros(masks[0].shape, dtype=np.uint8)
        for m in masks:
            mask = np.bitwise_or(mask, m)
        mask = 255 - mask
        
        fg = np.asarray(fg_mask)
        
        mask = np.bitwise_and(mask, fg)
        print(mask.dtype)
        
        mask = Image.fromarray(mask).convert("L")
        # new_photo = mask
        
        # Generate new person based on reference photo.
        person = generator.generate_person(
            prompt=str(Prompts.GENERATE_PERSON) + ", " + prompt, 
            negative_prompt=str(Prompts.GENERATE_PERSON_NEGATIVE) + ", " + negative_prompt,
            image_prompt=reference_photo,
            width=width,
            height=height
        )

        # composite = photo.copy()
        # composite.paste(person, (x,y))
        
        # # Inpaint person on image.
        # new_photo = inpaintor.inject_person(
        #     image=photo,
        #     person=person,
        #     mask=mask,
        #     prompt=str(Prompts.INPAINT_PERSON) + ", " + prompt,
        #     negative_prompt=str(Prompts.INPAINT_PERSON_NEGATIVE) + ", " + negative_prompt
        # )

        # Convert and return result.
        img_byte_arr = io.BytesIO()
        mask.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.exception("An error occurred") 
        raise HTTPException(status_code=500, detail="Internal server error.")