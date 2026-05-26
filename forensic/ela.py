import os
import cv2
import numpy as np
from PIL import Image, ImageChops
from typing import Tuple

def compute_ela(image_path: str, quality: int = 90, scale_factor: int = 25) -> Tuple[np.ndarray, float]:
    """Computes Error Level Analysis (ELA) on the target image.

    Args:
        image_path: Absolute path to the source image.
        quality: JPEG compression quality to resave (typically 90-95).
        scale_factor: Brightness multiplier for the output ELA map.

    Returns:
        Tuple[ela_map, forensic_score]: ELA image as a NumPy array (RGB) and a manipulation score.
    """
    temp_filename = image_path + ".resaved.jpg"
    
    try:
        # Load image via PIL to match JPEG saving pipelines precisely
        original = Image.open(image_path).convert('RGB')
        
        # Save image as JPEG at specified quality
        original.save(temp_filename, 'JPEG', quality=quality)
        
        # Open resaved image
        resaved = Image.open(temp_filename)
        
        # Compute absolute difference between original and resaved images
        diff = ImageChops.difference(original, resaved)
        
        # Find extrema to check if there is any difference at all
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
            
        # Scale difference for visualization
        scale = 255.0 / max_diff
        # Apply scaling and convert to numpy array
        ela_img = ImageEnhance_brightness_scale(diff, scale_factor)
        ela_np = np.array(ela_img)
        
        # Calculate a simple forensic score: average intensity of differences
        # High score (> 15.0) indicate high localized variance in compression quality, suggesting edit/forgery
        forensic_score = float(np.mean(np.array(diff)))
        
        return ela_np, forensic_score
        
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass

def ImageEnhance_brightness_scale(diff_img: Image.Image, scale: int) -> Image.Image:
    """Enhances difference image pixels for visualization."""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(diff_img)
    return enhancer.enhance(scale)
