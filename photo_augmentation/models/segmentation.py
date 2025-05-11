from typing import Union, List, Tuple

import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


class PersonSegmentationModel:
    """A model for segmenting persons in images using Segment Anything Model (SAM).
    
    This class provides an interface to SAM for generating high-quality segmentation
    masks for persons given their bounding boxes.
    """

    def __init__(self, model_type="vit_b", checkpoint="./sam_vit_b_01ec64.pth"):
        """Initialize the segmentation model with SAM.
        
        Parameters
        ----------
        model_type : str, optional
            The type of SAM model to use (e.g., 'vit_b', 'vit_l', 'vit_h'),
            by default "vit_b"
        checkpoint : str, optional
            Path to the pretrained model checkpoint file,
            by default "./sam_vit_b_01ec64.pth"
            
        Notes
        -----
        - The model type must match the checkpoint file
        - Different model types offer different tradeoffs between speed and accuracy
        """

        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.predictor = SamPredictor(self.model)
    

    def segment(self, image: Union[np.ndarray, Image.Image], boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Generate segmentation masks for persons given their bounding boxes.
        
        Parameters
        ----------
        image : Union[np.ndarray, Image.Image]
            Input image containing persons to segment
        boxes : List[Tuple[int, int, int, int]]
            List of bounding boxes for persons, where each box is represented as
            (x1, y1, x2, y2) coordinates of the top-left and bottom-right corners
            
        Returns
        -------
        np.ndarray
            Array of binary segmentation masks for each input box, with shape
            (N, H, W) where N is the number of boxes, and values are 0 or 255
            
        Notes
        -----
        - The input image is automatically converted to numpy array if needed
        - Each output mask is a binary mask where 255 indicates the person region
        - The masks will have the same height and width as the input image
        - For best results, the bounding boxes should tightly enclose the persons
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
        
        masks = np.asarray(masks, dtype=np.uint8) * 255
        return masks
