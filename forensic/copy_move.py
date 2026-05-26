import cv2
import numpy as np
from typing import Tuple

def detect_copy_move(image_path: str, min_matches: int = 4, distance_threshold: float = 0.7) -> Tuple[np.ndarray, bool]:
    """Detects copy-move forgery in an image by searching for identical keypoint clusters.

    Args:
        image_path: Absolute path to the image file.
        min_matches: Minimum keypoint matches required to flag copy-move.
        distance_threshold: Ratio test threshold for matching.

    Returns:
        Tuple[output_image, copy_move_detected]: Visualised matching links and boolean flag.
    """
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8), False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try SIFT first, fallback to ORB if not compiled
    detector = None
    try:
        detector = cv2.SIFT_create()
    except AttributeError:
        detector = cv2.ORB_create(nfeatures=1000)

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    if descriptors is None or len(keypoints) < 10:
        return img, False

    # Match descriptors against themselves
    bf = cv2.BFMatcher(cv2.NORM_L2 if hasattr(cv2, 'SIFT_create') else cv2.NORM_HAMMING)
    
    # K-Nearest Neighbors search to find self-similar blocks
    matches = bf.knnMatch(descriptors, descriptors, k=3)
    
    good_matches = []
    for m in matches:
        if len(m) < 3:
            continue
        # m[0] is the keypoint matching itself (distance = 0)
        # m[1] is the nearest distinct keypoint
        # m[2] is the second nearest distinct keypoint
        first_match = m[1]
        
        # Prevent matching keypoints that are physically adjacent (likely natural image textures)
        kp1 = keypoints[first_match.queryIdx]
        kp2 = keypoints[first_match.trainIdx]
        dist_px = np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + (kp1.pt[1] - kp2.pt[1])**2)
        
        if dist_px > 30.0: # Minimum pixel separation
            # Apply ratio test relative to 2nd match
            if first_match.distance < distance_threshold * m[2].distance:
                good_matches.append(first_match)

    copy_move_detected = len(good_matches) >= min_matches
    
    # Draw matched links onto original image
    out_img = img.copy()
    if copy_move_detected:
        for match in good_matches:
            pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
            pt2 = tuple(map(int, keypoints[match.trainIdx].pt))
            cv2.line(out_img, pt1, pt2, (0, 0, 255), 2)
            cv2.circle(out_img, pt1, 4, (0, 255, 0), -1)
            cv2.circle(out_img, pt2, 4, (0, 255, 0), -1)

    return out_img, copy_move_detected
