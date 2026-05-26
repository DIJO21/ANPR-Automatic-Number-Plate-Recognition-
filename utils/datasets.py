import os
import cv2
import logging
import shutil
import kagglehub
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ForensicANPR.Datasets")

def setup_kaggle_credentials():
    """Sets the provided Kaggle credentials in the environment variables."""
    os.environ['KAGGLE_USERNAME'] = "digil"
    os.environ['KAGGLE_KEY'] = "KGAT_66a219eed4d239d1704050c155846363"
    logger.info("Kaggle credentials configured in environment.")

def download_forensic_datasets() -> dict:
    """Downloads the required road-crossing and IDD datasets via kagglehub."""
    setup_kaggle_credentials()
    paths = {}
    
    try:
        logger.info("Initiating download: road-crossing-dataset...")
        path_road = kagglehub.dataset_download("siddhi17/road-crossing-dataset")
        logger.info(f"Road Crossing Dataset downloaded to: {path_road}")
        paths["road_crossing"] = path_road
    except Exception as e:
        logger.error(f"Failed to download road-crossing-dataset: {str(e)}")
        paths["road_crossing"] = None

    try:
        logger.info("Initiating download: new-idd-dataset...")
        path_idd = kagglehub.dataset_download("mitanshuchakrawarty/new-idd-dataset")
        logger.info(f"New IDD Dataset downloaded to: {path_idd}")
        paths["idd"] = path_idd
    except Exception as e:
        logger.error(f"Failed to download new-idd-dataset: {str(e)}")
        paths["idd"] = None

    return paths

def validate_image_file(image_path: Path) -> bool:
    """Forensically validates if an image is corrupted or readable by PIL/OpenCV."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Additional load check to test decompression
        cv_img = cv2.imread(str(image_path))
        if cv_img is None:
            return False
        return True
    except Exception:
        return False

def clean_corrupted_images(directory_path: str) -> int:
    """Scans a directory recursively and removes corrupted or invalid images."""
    dir_path = Path(directory_path)
    removed_count = 0
    
    if not dir_path.exists():
        logger.warning(f"Directory {directory_path} does not exist. Skipping cleaning.")
        return 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for file_path in dir_path.rglob("*"):
        if file_path.suffix.lower() in image_extensions:
            if not validate_image_file(file_path):
                logger.warning(f"Removing corrupted image: {file_path}")
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
                    
    logger.info(f"Completed image cleaning. Removed {removed_count} corrupted images.")
    return removed_count

def generate_mock_anpr_dataset(root_dir: str = "datasets/license_plates"):
    """Generates a small mock dataset of plates for testing when real downloads are unavailable."""
    root = Path(root_dir)
    for split in ["train", "val"]:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Generate a dummy canvas image containing a mock license plate
        for idx in range(5):
            img_path = img_dir / f"mock_plate_{idx}.jpg"
            lbl_path = lbl_dir / f"mock_plate_{idx}.txt"

            # Create standard white/black license plate canvas image
            canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 128  # gray background
            # Add a white box for license plate
            cv2.rectangle(canvas, (200, 200), (440, 280), (255, 255, 255), -1)
            cv2.rectangle(canvas, (200, 200), (440, 280), (0, 0, 0), 2)
            cv2.putText(canvas, "DL 3C AY 1111", (210, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            cv2.imwrite(str(img_path), canvas)

            # YOLO coordinates for plate: class_id, x_center, y_center, width, height (normalized)
            # Box is at x1=200, y1=200, x2=440, y2=280 inside 640x480 canvas
            x_center = (200 + 440) / 2.0 / 640.0
            y_center = (200 + 280) / 2.0 / 480.0
            width = (440 - 200) / 640.0
            height = (280 - 200) / 480.0

            with open(lbl_path, "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    logger.info(f"Mock license plate dataset generated at {root.resolve()}.")

import numpy as np # Ensure numpy is imported for canvas creation
