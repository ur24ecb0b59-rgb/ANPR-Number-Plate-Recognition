"""
detector.py
-----------
Stage 1: License plate detection using YOLOv8.
Returns bounding boxes of detected plates from an image.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class Detection:
    """Represents a single detected license plate region."""
    bbox: tuple          # (x1, y1, x2, y2)
    confidence: float
    crop: np.ndarray     # cropped plate image


class PlateDetector:
    """
    Detects license plates in images using YOLOv8.

    Args:
        model_path: Path to a custom-trained YOLOv8 weights file (.pt).
                    Defaults to the pretrained nano model as a starting point.
        conf_threshold: Minimum detection confidence (0–1).
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        device: str = "cpu",
    ):
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            )
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR image (as returned by cv2.imread).

        Returns:
            List of Detection objects, one per plate found.
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                crop = image[y1:y2, x1:x2]
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    crop=crop,
                ))

        return detections

    def detect_from_file(self, image_path: str) -> List[Detection]:
        """Load an image from disk and run detection."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.detect(image)


# ── Fallback: OpenCV Haar Cascade detector (no deep learning required) ────────

class HaarPlateDetector:
    """
    Lightweight fallback detector using OpenCV's Haar cascade.
    Works without GPU or large model files.
    """

    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 4):
        self.detector = cv2.CascadeClassifier(self.CASCADE_PATH)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, image: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plates = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
        )
        detections = []
        for (x, y, w, h) in plates:
            crop = image[y : y + h, x : x + w]
            detections.append(Detection(
                bbox=(x, y, x + w, y + h),
                confidence=1.0,
                crop=crop,
            ))
        return detections

    def detect_from_file(self, image_path: str) -> List[Detection]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.detect(image)
