from typing import List, Tuple, Union

import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from skimage.morphology import closing, opening


class PersonSegmentationModel:
    """
    A model for segmenting persons in images using Segment Anything Model (SAM).
    
    This class provides an interface to SAM for generating high-quality segmentation
    masks for persons given their bounding boxes.
    """

    def __init__(self, model_type="vit_b", checkpoint="./models/sam_vit_b_01ec64.pth"):
        """
        Initialize the segmentation model with SAM.

        :param model_type: The type of SAM model to use.
        :param checkpoint: Path to the pretrained model checkpoint file.
        """

        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.predictor = SamPredictor(self.model)
    
    
    def __postprocess_masks(self, masks: List[np.ndarray], kernel_size: int=7) -> List[np.ndarray]:
        new_masks = []
        for mask in masks:
            mask = closing(mask, np.ones((kernel_size, kernel_size)))
            mask = opening(mask, np.ones((kernel_size, kernel_size)))
            new_masks.append(mask)
            
        return new_masks


    def segment(self, image: Union[np.ndarray, Image.Image], boxes: List[Tuple[int, int, int, int]]) \
        -> np.ndarray:
        """
        Generate segmentation masks for persons given their bounding boxes.

        :param image: Input image containing persons to segment.
        :param boxes: List of bounding boxes for persons.
        :returns: Array of binary segmentation masks for each input box.
        """

        image = np.asarray(image)
        self.predictor.set_image(image)
        
        masks = []
        for box in boxes:
            input_box = np.array(box)
            mask, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            masks.append(mask[0])

        masks = self.__postprocess_masks(masks)
        
        masks = np.asarray(masks, dtype=np.uint8) * 255
        return masks
