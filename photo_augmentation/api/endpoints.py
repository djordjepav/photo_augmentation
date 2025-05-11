
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import io
from PIL import Image, ImageDraw
import numpy as np
import logging

from . import app
from ..models.diffusion import PersonGeneratingModel, PhotoInpaintingModel
from ..models.detection import PersonDetectionModel
from ..models.segmentation import PersonSegmentationModel

from ..models.utils import crop_to_divisible


BASE_PROMPT = "A stunningly realistic portrait of a young woman, \
    detailed facial features, soft natural lighting, intricate skin \
    texture with pores, realistic eyes with reflections, symmetrical face, \
    delicate blush, glossy lips, soft bokeh background, ultra HD, 8K, \
    professional photography, Canon EOS R5, 85mm lens, photorealistic, \
    subtle skin imperfections, volumetric lighting"
    
BASE_NEGATIVE_PROMPT = "blurry, deformed eyes, bad anatomy, extra limbs, \
    disfigured fingers, distorted face, poorly drawn hands, unnatural skin tone,\
    overexposed, underexposed, lowres, bad proportions, long neck, cloned features, \
    watermark, text, cartoon, painting, 3D render, doll-like, plastic skin, mutated, \
    extra arms, malformed limbs"

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
    reference_photo_scale:float =0.7,    
):
    try:
        prompt = BASE_PROMPT + ", " + prompt
        negative_prompt = BASE_NEGATIVE_PROMPT + ", " + negative_prompt
        
        reference_photo = Image.open(io.BytesIO(await reference_photo.read()))
        
        person = generator.generate_person(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            image_prompt=reference_photo,
            ip_adapter_scale=reference_photo_scale,
            width=768,
            height=768
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
    reference_photo_scale:float =0.7,
    prompt: str = "a photo of a person",
    negative_prompt: str = "",
    person_index: int = 0,
):
    try:
        prompt = BASE_PROMPT + ", " + prompt
        negative_prompt = BASE_NEGATIVE_PROMPT + ", " + negative_prompt

        photo = Image.open(io.BytesIO(await photo.read()))
        photo = crop_to_divisible(photo)
        
        reference_photo = Image.open(io.BytesIO(await reference_photo.read()))
        reference_photo = crop_to_divisible(reference_photo)
        
        bboxes = detector.detect(photo)        
        masks = segmentor.segment(photo, bboxes)
        
        mask = masks[person_index]
        # new_photo = Image.fromarray(mask)
        
        print(mask.max())
        
        print("Segmentation performed, num people:", len(masks))
        
        person = generator.generate_person(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            image_prompt=reference_photo,
            ip_adapter_scale=reference_photo_scale,
            width=768,
            height=768
        )
        
        print("Person generated!")
        
        new_photo = inpaintor.inpaint_person(
            image=photo,
            person=person,
            mask=mask
        )

        img_byte_arr = io.BytesIO()
        new_photo.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
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
        photo = crop_to_divisible(photo)

        reference_photo = Image.open(io.BytesIO(await reference_photo.read()))
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
        
        # Merge masks and invert them.
        mask = np.zeros(masks[0].shape, dtype=np.uint8)
        for m in masks:
            mask = np.bitwise_or(mask, m)
        mask = 255 - mask
        
        fg = np.asarray(fg_mask)
        
        mask = np.bitwise_and(mask, fg)
        print(mask.dtype)
        
        mask = Image.fromarray(mask).convert("L")
        new_photo = mask

        # Append base prompts.
        prompt = BASE_PROMPT + ", " + prompt
        negative_prompt = BASE_NEGATIVE_PROMPT + ", " + negative_prompt
        
        # Generate new person based on reference photo.
        person = generator.generate_person(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            image_prompt=reference_photo,
            width=128,
            height=256
        )

        composite = photo.copy()
        composite.paste(person, (x,y))
        
        # # Inpaint person on image.
        # new_photo = inpaintor.inject_person(
        #     image=composite,
        #     person=person,
        #     mask=mask,
        #     prompt=prompt,
        #     negative_prompt=negative_prompt
        # )

        # Convert and return result.
        img_byte_arr = io.BytesIO()
        composite.save(img_byte_arr, format='PNG')
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
        image = crop_to_divisible(image)

        mask = np.zeros(image.size[::-1], dtype=np.uint8)
        print(mask.shape)
        mask[x:x+width, y:y+height] = 255
        
        # Inpaint masked region.
        group_image = inpaintor.inpaint_region(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt
        )

        # Return result
        img_byte_arr = io.BytesIO()
        group_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.exception("An error occurred") 
        raise HTTPException(status_code=500, detail="Internal server error.")
