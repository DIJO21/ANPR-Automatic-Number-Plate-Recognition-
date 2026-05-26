import os
import shutil
import uuid
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from inference import ForensicANPRPipeline
from utils.gpu import get_gpu_info

app = FastAPI(
    title="Forensic ANPR API Gateway",
    description="REST API for license plate recognition, tracking, and image forgery forensics.",
    version="1.0.0"
)

# Active Watchlist and pipeline instantiation
WATCHLIST = ["DL3CAY1111", "MH12GP1234", "KA51MB8899"]
if os.path.exists("weights/best.pt"):
    yolo_weights = "weights/best.pt"
elif os.path.exists("best.pt"):
    yolo_weights = "best.pt"
else:
    yolo_weights = "yolov8n.pt"
pipeline = ForensicANPRPipeline(yolo_weights=yolo_weights, ocr_weights=None, watchlist=WATCHLIST)

# Setup temporary upload directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

class WatchlistItem(BaseModel):
    plate_number: str

@app.get("/")
def read_root():
    """Returns server diagnostics and GPU capabilities."""
    gpu_stats = get_gpu_info()
    return {
        "status": "online",
        "service": "Forensic ANPR & Image Tampering Analysis Engine",
        "active_gpu": gpu_stats["current_device_name"] if gpu_stats["cuda_available"] else "CPU fallback mode",
        "allocated_vram_mb": f"{gpu_stats['allocated_mb']:.2f}"
    }

@app.get("/watchlist", response_model=List[str])
def get_watchlist():
    """Retrieves the active stolen/wanted vehicle plate watchlist."""
    return WATCHLIST

@app.post("/watchlist")
def add_to_watchlist(item: WatchlistItem):
    """Adds a new plate string to the active monitoring watchlist."""
    plate_std = item.plate_number.replace(" ", "").upper()
    if plate_std not in WATCHLIST:
        WATCHLIST.append(plate_std)
        # Re-initialize watchlist matcher in pipeline
        pipeline.watchlist_matcher._init_faiss()
        return {"message": f"Plate {plate_std} added to watchlist successfully."}
    return {"message": f"Plate {plate_std} is already present on watchlist."}

@app.post("/predict")
async def predict_plate(file: UploadFile = File(...)):
    """Processes an uploaded image for plate recognition and digital tampering markers."""
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG images are supported.")

    file_extension = Path(file.filename).suffix
    unique_id = str(uuid.uuid4())
    temp_input_path = UPLOAD_DIR / f"{unique_id}{file_extension}"
    
    try:
        # Save file to temp upload location
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run pipeline
        res = pipeline.process_image(str(temp_input_path))
        
        # Save annotated image
        annotated_filename = f"annotated_{unique_id}.jpg"
        annotated_path = OUTPUT_DIR / annotated_filename
        cv2.imwrite(str(annotated_path), res["annotated_image"])

        # Format output payload
        plates = []
        for plate in res["plates_detected"]:
            plates.append({
                "box": plate["box"],
                "confidence": plate["score"],
                "ocr": plate["corrected_ocr"],
                "raw_ocr": plate["raw_ocr"],
                "rto_info": plate["rto_info"],
                "watchlist_status": plate["watchlist_status"]
            })

        return {
            "plates": plates,
            "forensics": {
                "ela_score": res["forensics"]["ela_score"],
                "exif_tampered": res["forensics"]["exif_tampered"],
                "exif_details": res["forensics"]["exif_details"],
                "double_jpeg_detected": res["forensics"]["double_jpeg_detected"],
                "copy_move_detected": res["forensics"]["copy_move_detected"],
                "is_manipulated": res["forensics"]["is_manipulated"]
            },
            "annotated_image_url": f"/download/{annotated_filename}"
        }

    except Exception as e:
        logger.error(f"Inference API call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal inference failure: {str(e)}")
        
    finally:
        # Delete original uploaded file to conserve space
        if temp_input_path.exists():
            temp_input_path.unlink()

@app.get("/download/{filename}")
def download_file(filename: str):
    """Serves the annotated forensic results file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Requested file not found.")
    return FileResponse(str(file_path))

from pathlib import Path
