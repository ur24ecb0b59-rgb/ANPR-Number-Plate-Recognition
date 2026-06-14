"""
ocr.py
------
Stage 2: Extract text from a cropped license plate image using EasyOCR.
Includes preprocessing to improve recognition accuracy.
"""

import cv2
import numpy as np
import re
from dataclasses import dataclass
from typing import Optional

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


@dataclass
class OCRResult:
    """Holds the raw and cleaned text extracted from a plate crop."""
    raw_text: str
    plate_text: str      # cleaned / formatted
    confidence: float


def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """
    Apply a series of image processing steps to improve OCR accuracy:
    1. Resize to fixed height while preserving aspect ratio.
    2. Convert to grayscale.
    3. Apply adaptive thresholding (handles uneven lighting).
    4. Denoise.
    """
    # 1. Resize
    target_h = 64
    h, w = crop.shape[:2]
    scale = target_h / h
    resized = cv2.resize(crop, (int(w * scale), target_h))

    # 2. Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # 4. Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)

    return denoised


def clean_plate_text(raw: str) -> str:
    """
    Remove unwanted characters and normalize the plate string.
    Keeps only alphanumeric characters and hyphens.
    """
    cleaned = re.sub(r"[^A-Z0-9\-]", "", raw.upper().strip())
    return cleaned


class PlateOCR:
    """
    Reads text from a cropped plate image using EasyOCR.

    Args:
        lang_list: Languages to recognise. Default is English.
        gpu: Use GPU acceleration if available.
    """

    def __init__(self, lang_list: list = None, gpu: bool = False):
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "easyocr is required. Install with: pip install easyocr"
            )
        self.reader = easyocr.Reader(lang_list or ["en"], gpu=gpu)

    def read(self, crop: np.ndarray) -> Optional[OCRResult]:
        """
        Run OCR on a single cropped plate image.

        Returns:
            OCRResult with raw text, cleaned plate text, and confidence.
            Returns None if no text is detected.
        """
        processed = preprocess_plate(crop)
        results = self.reader.readtext(processed, detail=1, paragraph=False)

        if not results:
            return None

        # Concatenate all detected text segments
        raw_parts = [r[1] for r in results]
        confs = [r[2] for r in results]

        raw_text = " ".join(raw_parts)
        avg_conf = sum(confs) / len(confs)
        plate_text = clean_plate_text(raw_text)

        return OCRResult(
            raw_text=raw_text,
            plate_text=plate_text,
            confidence=round(avg_conf, 4),
        )
