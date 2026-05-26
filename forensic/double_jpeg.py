import cv2
import numpy as np

def detect_double_jpeg(image_path: str) -> bool:
    """Detects double JPEG compression artifacts in an image by calculating blocking grid metrics.

    When an image is saved as JPEG, block compression creates a subtle 8x8 grid. If edited
    and resaved, the grids misalign, creating distinct pixel gradient variances.

    Args:
        image_path: Absolute path to the JPEG image file.

    Returns:
        Boolean indicating if double JPEG compression is detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    if h < 16 or w < 16:
        return False

    # Compute differences across 8x8 block boundaries
    # We compare differences at boundary columns/rows vs internal ones.
    
    # Extract columns at block boundary (e.g., indices 7, 15, 23...) vs internal columns
    boundary_diff_h = 0.0
    internal_diff_h = 0.0
    
    for x in range(8, w - 8, 8):
        diff_boundary = np.abs(gray[:, x].astype(np.float32) - gray[:, x - 1].astype(np.float32))
        diff_internal = np.abs(gray[:, x + 3].astype(np.float32) - gray[:, x + 2].astype(np.float32))
        boundary_diff_h += np.mean(diff_boundary)
        internal_diff_h += np.mean(diff_internal)
        
    boundary_diff_v = 0.0
    internal_diff_v = 0.0
    for y in range(8, h - 8, 8):
        diff_boundary = np.abs(gray[y, :].astype(np.float32) - gray[y - 1, :].astype(np.float32))
        diff_internal = np.abs(gray[y + 3, :].astype(np.float32) - gray[y + 2, :].astype(np.float32))
        boundary_diff_v += np.mean(diff_boundary)
        internal_diff_v += np.mean(diff_internal)

    # Average ratios
    num_blocks_h = (w - 16) // 8
    num_blocks_v = (h - 16) // 8
    
    if num_blocks_h <= 0 or num_blocks_v <= 0:
        return False
        
    avg_boundary_h = boundary_diff_h / num_blocks_h
    avg_internal_h = internal_diff_h / num_blocks_h
    
    avg_boundary_v = boundary_diff_v / num_blocks_v
    avg_internal_v = internal_diff_v / num_blocks_v

    ratio_h = avg_boundary_h / (avg_internal_h + 1e-5)
    ratio_v = avg_boundary_v / (avg_internal_v + 1e-5)

    # If the ratio is very close to 1.0 or less, or highly deviant, it suggests
    # double JPEG compression (since single JPEGs have elevated boundary gradients due to blockiness,
    # whereas double compression flattens or shifts this distribution).
    # Forensic baseline: single compression ratio is typically > 1.15. Ratios below 1.05 suggest double compression.
    is_double_compressed = (ratio_h < 1.04) or (ratio_v < 1.04)
    
    return bool(is_double_compressed)
