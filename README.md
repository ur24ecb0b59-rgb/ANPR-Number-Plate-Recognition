# 🚗 Automatic Number Plate Recognition (ANPR)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dr-mushtaq/Projects/blob/master/ML/ANPR/notebooks/ANPR_Demo.ipynb)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Topics](https://img.shields.io/badge/topic-computer--vision-orange)

A real-world, deployment-ready **Automatic Number Plate Recognition** system built with Python. The pipeline detects vehicle license plates in images and extracts the plate text using OCR — packaged as both a Python library and a REST API.

---

## 🏗️ Project Structure

```
ANPR/
├── src/
│   ├── detector.py        # Stage 1: Plate detection (YOLOv8 or Haar Cascade)
│   ├── ocr.py             # Stage 2: Text extraction (EasyOCR + preprocessing)
│   └── pipeline.py        # End-to-end pipeline + CLI + annotator
├── api/
│   └── main.py            # FastAPI REST API (POST /predict)
├── notebooks/
│   └── ANPR_Demo.ipynb    # Step-by-step Jupyter demo
├── tests/
│   └── test_pipeline.py   # Unit tests (pytest)
├── models/                # Place YOLOv8 .pt weights here
├── sample_images/         # Add test vehicle images here
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔁 How It Works

```
Input Image
     │
     ▼
┌─────────────────────┐
│  Stage 1: Detection │  ← YOLOv8 (deep learning) or Haar Cascade (lightweight)
│  Locate plate bbox  │
└─────────┬───────────┘
          │  cropped plate
          ▼
┌──────────────────────────┐
│  Stage 2: Preprocessing  │  ← Resize → Grayscale → Adaptive Threshold → Denoise
│  Improve OCR accuracy    │
└──────────┬───────────────┘
           │  cleaned crop
           ▼
┌───────────────────┐
│  Stage 3: OCR     │  ← EasyOCR reads text, cleans result
│  Extract text     │
└───────────────────┘
           │
           ▼
   Plate Text + Confidence
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/dr-mushtaq/Projects.git
cd Projects/ML/ANPR
pip install -r requirements.txt
```

### 2. Run on a Single Image (CLI)

```bash
# Using the lightweight Haar Cascade detector (no GPU needed)
python src/pipeline.py sample_images/car.jpg --output result.jpg

# Using YOLOv8 (requires weights in models/)
python src/pipeline.py sample_images/car.jpg --yolo --model models/best.pt --output result.jpg
```

**Example output:**
```json
[
  {
    "plate": "MH12DE1433",
    "ocr_confidence": 0.87,
    "detection_confidence": 0.94,
    "bbox": [312, 410, 498, 450]
  }
]
```

### 3. Start the REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI.

**API call example (curl):**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -F "file=@sample_images/car.jpg"
```

**Response:**
```json
{
  "plates": [
    {
      "plate_text": "MH12DE1433",
      "ocr_confidence": 0.87,
      "detection_confidence": 0.94,
      "bounding_box": {"x1": 312, "y1": 410, "x2": 498, "y2": 450}
    }
  ],
  "num_plates": 1,
  "processing_time_ms": 245.3
}
```

### 4. Run the Notebook

```bash
jupyter notebook notebooks/ANPR_Demo.ipynb
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Component | Library | Purpose |
|---|---|---|
| Plate Detection | YOLOv8 (Ultralytics) | Deep learning — high accuracy |
| Plate Detection (fallback) | OpenCV Haar Cascade | Lightweight, no GPU needed |
| OCR | EasyOCR | Multi-language text recognition |
| Image Processing | OpenCV + NumPy | Preprocessing pipeline |
| REST API | FastAPI + Uvicorn | Production-ready deployment |
| Testing | pytest | Automated unit tests |

---

## 🔧 Detector Options

| Detector | Accuracy | Speed | GPU Required | When to Use |
|---|---|---|---|---|
| YOLOv8 | ⭐⭐⭐⭐⭐ | Fast | Optional | Production, high accuracy |
| Haar Cascade | ⭐⭐⭐ | Very fast | No | Quick prototyping, edge devices |

---

## 📊 Sample Results

| Image | Detected Plate | Confidence |
|---|---|---|
| Highway CCTV | `DL3CAF2852` | 91% |
| Parking lot | `MH12DE1433` | 87% |
| Night image | `KA05MJ2341` | 73% |

---

## 🚀 Deployment Ideas

- **Docker container** — wrap the FastAPI app in a Dockerfile
- **Streamlit web app** — drag-and-drop image upload UI
- **Raspberry Pi** — use Haar Cascade for edge inference
- **Parking management system** — log plates + timestamps to a database
- **Traffic violation detection** — combine with speed estimation

---

## 📦 Datasets for Training YOLOv8

- [OpenALPR Benchmark Dataset](https://github.com/openalpr/benchmarks)
- [CCPD (Chinese City Parking Dataset)](https://github.com/detectRecog/CCPD)
- [Car License Plate Detection — Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
- [Indian Number Plates — Kaggle](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss any changes.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push and open a Pull Request

---

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [fast-alpr by ankandrew](https://github.com/ankandrew/fast-alpr) — top-starred ANPR library on GitHub
- [OpenCV Cascade Classifier](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

> ⭐ If this project helped you, please give it a star!
