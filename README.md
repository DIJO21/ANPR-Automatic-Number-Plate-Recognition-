
# Forensic ANPR Ecosystem (Indian Plate Focus)

An enterprise-grade, modular Automatic Number Plate Recognition (ANPR) and digital image forensics platform. The system is designed to identify vehicle plates, validate registration jurisdictions based on Indian Regional Transport Office (RTO) standards, check plates against criminal watchlists, and perform pixel-level image forgery checks (Error Level Analysis, EXIF metadata tags, copy-move detection, and double JPEG grid compression).

---

## Key Features

1. **Precision License Plate Localization:** Powered by Ultralytics YOLOv8/v11.
2. **Interactive Cropper & Video Timeline Extractor:** Includes a Streamlit crop interface immediately after upload, and frame-by-frame seeking with plate cropping on video timelines.
3. **Indian License Plate Rules Engine:**
   * **Standard private & transport formats:** `ST CC XX ####` (e.g. `MH 12 GP 1234`, `DL 3C AY 1111`)
   * **Bharat Series (BH):** `YY BH #### XX`
   * **Military registration format:** Broad upward arrow prefixed year codes `↑YYA######X`
   * **Temporary trade formats & Diplomatic/Consular formats**
   * **State codes database:** Automatic RTO jurisdiction lookup mapping plates to states.
4. **Spatial Concatenator for Multi-Row Plates:** Detects and sorts split lines of text (e.g. motorcycle plates with `KL 08` on top and `AH1509` below) into correct sequential order.
5. **OCR Candidate Variation Generator:** Dynamically permutes commonly confused letters/numbers (e.g. `0` ↔ `O` ↔ `D`, `8` ↔ `B`), validates them against RTO patterns, and displays a comprehensive alternatives grid.
6. **Digital Image Forensics Suite:**
   * **Error Level Analysis (ELA):** Detects differences in quantization levels across regions.
   * **EXIF Metadata Integrity Check:** Inspects tags for image editing tools (Photoshop, Canva, Snapseed, GIMP).
   * **Copy-Move Duplication Matching:** Identifies cloned plate numbers using self-matching descriptors.
   * **Double JPEG Compression Checker:** Checks periodic anomalies in 8x8 blocking grids.
7. **PDF Forensic Report Compiler:** Compiles certified PDF reports with signature boxes, evidence charts, and integrity hashes.

---

## Directory Architecture

```text
forensic_anpr_system/
│
├── app.py                  # Streamlit Forensics Investigator UI Dashboard
├── requirements.txt        # Deep learning, vision, and web package dependencies
├── setup.sh                # Automation script to prepare environment folder structures
├── train_yolo.py           # YOLOv8/v11 GPU training pipeline with Cosine LR schedulers
├── train_ocr.py            # PyTorch CRNN+CTC character recognition training script
├── export_models.py        # Model ONNX & TensorRT exporter helper
├── inference.py            # Main pipeline runner (frame batching & tracker)
├── api_server.py           # FastAPI REST endpoints for remote cameras
├── Dockerfile              # Containerization configuration
│
├── configs/
│   └── yolo_config.yaml    # YOLO datasets path specifications
│
├── forensic/               # Forgery analysis calculators
│   ├── ela.py
│   ├── exif.py
│   ├── copy_move.py
│   └── double_jpeg.py
│
├── decoder/                # Spell checking & Indian plate syntax rules
│   ├── indian_plates.py
│   └── probabilistic.py
│
├── utils/                  # GPU monitoring, report generators, and tracking helpers
│   ├── gpu.py
│   ├── tracking.py
│   ├── datasets.py
│   └── report_generator.py
│
├── notebooks/
│   └── forensic_anpr_colab.ipynb # Fully formatted Google Colab GPU training notebook
│
└── tests/                  # Pytest unit testing suite
    ├── test_decoder.py
    └── test_forensics.py
```

---

## Quick Start Guide

### 1. Initialize Folders and Dependencies
Run the setup shell script to configure directory hierarchies and install dependencies:
```bash
bash setup.sh
```
Or manually install dependencies:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> **PaddlePaddle CPU Execution Troubleshoot:** If you run on CPU and PaddlePaddle throws instruction compiler errors, disable oneDNN support by setting these environment variables before running your script:
> ```python
> import os
> os.environ["FLAGS_use_onednn"] = "0"
> os.environ["FLAGS_enable_onednn_gqa_pass"] = "0"
> ```
> This fix is already integrated into `inference.py`.

### 2. Launch the Streamlit Dashboard UI
```bash
streamlit run app.py
```
Open `http://localhost:8501` to access the premium dark-mode forensic investigator panel.

### 3. Spin up the FastAPI Server Gateway
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/docs` to inspect active REST endpoints.

### 4. Running Docker Containers
Build and run the entire ecosystem using containerization:
```bash
docker build -t forensic-anpr .
docker run -p 8501:8501 -p 8000:8000 forensic-anpr
```

---

## Model Training

### YOLOv8/v11 Plate Detection
To train YOLO on a GPU-enabled machine or Google Colab, execute:
```bash
python train_yolo.py --model yolov8n.pt --epochs 50
```
*Specifying `--colab_drive /content/drive/MyDrive` will automatically sync epoch weights directly to Google Drive to ensure resilience against session timeouts.*

### CRNN Scene Text Recognition
To train/fine-tune the character recognition model using synthetic RTO generator batches:
```bash
python train_ocr.py --epochs 20 --dataset_size 5000
```

### ONNX Exporting
Export trained model weights to ONNX format:
```bash
python export_models.py --yolo_path checkpoints/yolo/best.pt --ocr_path checkpoints/ocr/ocr_crnn_final.pth
```

---

## Running Unit Tests
To run the automated validation tests for the Indian plate parser and digital forensic modules, execute:
```bash
python -m unittest discover -s tests
```
All tests are configured to use in-memory synthetic assets, making them executable without downloading external binary files.
=======
# ANPR-Automatic-Number-Plate-Recognition-