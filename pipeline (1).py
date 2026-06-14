"""
pipeline.py
-----------
End-to-end ANPR pipeline: detect plates → read text → annotate image.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from detector import PlateDetector, HaarPlateDetector, Detection
from ocr import PlateOCR, OCRResult


@dataclass
class PlateResult:
    """Final result for one detected plate."""
    bbox: tuple
    detection_conf: float
    plate_text: str
    ocr_conf: float
    crop: np.ndarray


class ANPRPipeline:
    """
    Full Automatic Number Plate Recognition pipeline.

    Args:
        use_yolo: Use YOLOv8 for detection (True) or Haar cascade (False).
        yolo_model: Path to YOLOv8 weights. Only used if use_yolo=True.
        det_conf: Detection confidence threshold.
        gpu: Use GPU for OCR.
    """

    def __init__(
        self,
        use_yolo: bool = False,
        yolo_model: str = "yolov8n.pt",
        det_conf: float = 0.4,
        gpu: bool = False,
    ):
        if use_yolo:
            self.detector = PlateDetector(model_path=yolo_model, conf_threshold=det_conf)
        else:
            self.detector = HaarPlateDetector()

        self.ocr = PlateOCR(gpu=gpu)

    def process_image(self, image: np.ndarray) -> List[PlateResult]:
        """
        Run the full pipeline on a BGR image.

        Returns:
            List of PlateResult objects (one per detected plate).
        """
        detections: List[Detection] = self.detector.detect(image)
        results: List[PlateResult] = []

        for det in detections:
            ocr_result: Optional[OCRResult] = self.ocr.read(det.crop)
            if ocr_result is None:
                continue
            results.append(PlateResult(
                bbox=det.bbox,
                detection_conf=det.confidence,
                plate_text=ocr_result.plate_text,
                ocr_conf=ocr_result.confidence,
                crop=det.crop,
            ))

        return results

    def process_file(self, image_path: str) -> List[PlateResult]:
        """Load image from disk and run the pipeline."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.process_image(image)

    def annotate(self, image: np.ndarray, results: List[PlateResult]) -> np.ndarray:
        """
        Draw bounding boxes and plate text on the image.

        Returns:
            Annotated copy of the image.
        """
        annotated = image.copy()
        for r in results:
            x1, y1, x2, y2 = r.bbox
            label = f"{r.plate_text}  ({r.ocr_conf:.0%})"

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # Label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw, y1), (0, 200, 0), -1)

            # Label text
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
            )

        return annotated


# ── Quick CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="ANPR Pipeline — single image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", default="output.jpg", help="Annotated output path")
    parser.add_argument("--yolo", action="store_true", help="Use YOLOv8 detector")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights path")
    args = parser.parse_args()

    pipeline = ANPRPipeline(use_yolo=args.yolo, yolo_model=args.model)
    image = cv2.imread(args.image)

    results = pipeline.process_image(image)
    annotated = pipeline.annotate(image, results)
    cv2.imwrite(args.output, annotated)

    output = [
        {
            "plate": r.plate_text,
            "ocr_confidence": r.ocr_conf,
            "detection_confidence": r.detection_conf,
            "bbox": r.bbox,
        }
        for r in results
    ]
    print(json.dumps(output, indent=2))
    print(f"\nAnnotated image saved to: {args.output}")
