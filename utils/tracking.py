import numpy as np
from typing import List, Dict, Tuple

class LicensePlateTracker:
    """A lightweight IOU-based Centroid Tracker to preserve plate ID consistency across video frames."""

    def __init__(self, max_disappeared: int = 15, min_iou: float = 0.3):
        self.next_object_id = 0
        self.objects: Dict[int, Tuple[int, int, int, int]] = {}  # ID -> (x1, y1, x2, y2)
        self.disappeared: Dict[int, int] = {}  # ID -> count of missed frames
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        self.plate_history: Dict[int, List[str]] = {}  # ID -> list of predicted OCRs across frames

    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y1_1, y2_2)  # Bug-check: y_bottom = min(y2_1, y2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = float(area1 + area2 - intersection_area)

        if union_area <= 0.0:
            return 0.0
        return intersection_area / union_area

    def register(self, box: Tuple[int, int, int, int]):
        """Registers a newly detected plate."""
        self.objects[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        self.plate_history[self.next_object_id] = []
        self.next_object_id += 1

    def deregister(self, object_id: int):
        """Removes a plate tracker ID."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, input_boxes: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        """Updates tracker states with new detections, mapping existing IDs based on IoU."""
        if len(input_boxes) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # If no objects are currently tracked, register all incoming detections
        if len(self.objects) == 0:
            for box in input_boxes:
                self.register(box)
            return self.objects

        # Compute IoU matrix between existing objects and incoming boxes
        object_ids = list(self.objects.keys())
        tracked_boxes = [self.objects[oid] for oid in object_ids]

        iou_matrix = np.zeros((len(tracked_boxes), len(input_boxes)), dtype=np.float32)
        for i, t_box in enumerate(tracked_boxes):
            for j, i_box in enumerate(input_boxes):
                iou_matrix[i, j] = self._calculate_iou(t_box, i_box)

        # Match based on highest IoU
        matched_objects = set()
        matched_detections = set()

        # Find maximum IoU matches iteratively
        while True:
            max_val = np.max(iou_matrix)
            if max_val < self.min_iou:
                break
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            # Record match
            oid = object_ids[i]
            self.objects[oid] = input_boxes[j]
            self.disappeared[oid] = 0
            matched_objects.add(oid)
            matched_detections.add(j)

            # Invalidate row and column
            iou_matrix[i, :] = -1.0
            iou_matrix[:, j] = -1.0

        # Mark unmatched objects as disappeared
        for oid in object_ids:
            if oid not in matched_objects:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        # Register unmatched incoming detections as new objects
        for j, box in enumerate(input_boxes):
            if j not in matched_detections:
                self.register(box)

        return self.objects

    def add_ocr_prediction(self, object_id: int, predicted_text: str):
        """Adds a predicted OCR text for temporal smoothing."""
        if object_id in self.plate_history:
            self.plate_history[object_id].append(predicted_text)

    def get_smoothed_ocr(self, object_id: int) -> str:
        """Determines the most probable OCR representation by analyzing temporal frequency."""
        if object_id not in self.plate_history or len(self.plate_history[object_id]) == 0:
            return ""
        
        predictions = self.plate_history[object_id]
        # Count the frequency of each unique prediction
        counts = {}
        for pred in predictions:
            counts[pred] = counts.get(pred, 0) + 1
        
        # Return the prediction with the highest frequency
        return max(counts, key=counts.get)
