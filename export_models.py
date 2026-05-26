import os
import argparse
import logging
import torch
from ultralytics import YOLO
from train_ocr import CRNN, ALPHABET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ForensicANPR.Export")

def export_yolo_onnx(model_path: str):
    """Exports a trained YOLO model to ONNX."""
    logger.info(f"Loading YOLO model for export: {model_path}")
    try:
        model = YOLO(model_path)
        logger.info("Exporting YOLO model to ONNX format...")
        # Ultralytics built-in ONNX export with default dynamic batch parameters
        export_path = model.export(format="onnx", dynamic=True)
        logger.info(f"YOLO exported successfully. Path: {export_path}")
    except Exception as e:
        logger.error(f"Failed to export YOLO model: {e}")

def export_ocr_onnx(model_path: str, output_path: str, nh: int = 256):
    """Exports the PyTorch CRNN model to ONNX format."""
    logger.info(f"Loading CRNN OCR model for export: {model_path}")
    device = torch.device("cpu")
    
    # Instantiate the CRNN model structure
    model = CRNN(img_h=32, nc=1, nclass=len(ALPHABET) + 1, nh=nh)
    
    # Load state dict
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Loaded CRNN model weights.")
        except Exception as e:
            logger.warning(f"Could not load custom weights ({e}). Exporting base initialization model.")
    else:
        logger.warning(f"Model path {model_path} not found. Exporting base initialization model.")

    model.eval()
    
    # Create dummy input matching: [batch, channel, height, width]
    dummy_input = torch.randn(1, 1, 32, 100, requires_grad=False)
    
    logger.info("Exporting CRNN model to ONNX format...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 3: 'width'},
                'output': {0: 'seq_len', 1: 'batch_size'}
            }
        )
        logger.info(f"CRNN exported successfully to ONNX. Path: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export CRNN model to ONNX: {e}")

def main():
    parser = argparse.ArgumentParser(description="Export Trained ANPR Models to ONNX")
    parser.add_argument("--yolo_path", type=str, default="yolov8n.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--ocr_path", type=str, default="checkpoints/ocr/ocr_crnn_final.pth", help="Path to CRNN weights (.pth)")
    parser.add_argument("--ocr_output", type=str, default="checkpoints/ocr/ocr_crnn.onnx", help="Path to save OCR ONNX model")
    parser.add_argument("--yolo_only", action="store_true", help="Only export YOLO model")
    parser.add_argument("--ocr_only", action="store_true", help="Only export OCR model")
    
    args = parser.parse_args()

    if not args.ocr_only:
        export_yolo_onnx(args.yolo_path)

    if not args.yolo_only:
        export_ocr_onnx(args.ocr_path, args.ocr_output)

if __name__ == "__main__":
    main()
