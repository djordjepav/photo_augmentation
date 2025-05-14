from typing import List, Tuple, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO


class PersonDetectionModel:
    """
    A person detection model using YOLOv8 for identifying and localizing persons in images.
    
    This class provides an interface to YOLOv8 object detection model specifically
    configured to detect persons in input images.
    """

    def __init__(self, weights: str="./models/yolov8n.pt"):
        """
        Initialize the person detection model with YOLOv8 nano weights.
        
        Loads the pre-trained YOLOv8n model ('yolov8n.pt') which is optimized
        for person detection tasks.
        """

        self.yolo_model = YOLO(weights)


    def detect(self, image: Union[np.ndarray, Image.Image]) -> List[Tuple[int, int, int, int]]:
        """
        Detect persons in the input image and return their bounding boxes.

        :param image: Input image to process.
        :returns: List of bounding boxes for detected persons.
        """

        image = np.asarray(image)
        
        results = self.yolo_model(image)
        person_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0:  # class 0 is person in YOLO
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    person_boxes.append((x1, y1, x2, y2))

        return person_boxes
