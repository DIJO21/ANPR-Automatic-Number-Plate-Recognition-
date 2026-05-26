import unittest
import os
import cv2
import numpy as np
from forensic.ela import compute_ela
from forensic.exif import ExifForensics
from forensic.copy_move import detect_copy_move
from forensic.double_jpeg import detect_double_jpeg

class TestForensics(unittest.TestCase):
    """Unit tests for digital image forgery detectors."""

    @classmethod
    def setUpClass(cls):
        # Create a dummy image for testing forensic calculators
        cls.test_img_path = "test_temp_canvas.jpg"
        canvas = np.zeros((200, 200, 3), dtype=np.uint8) + 128
        cv2.rectangle(canvas, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.imwrite(cls.test_img_path, canvas)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_img_path):
            os.remove(cls.test_img_path)

    def test_ela_computation(self):
        ela_map, score = compute_ela(self.test_img_path)
        self.assertIsNotNone(ela_map)
        self.assertEqual(ela_map.shape[:2], (200, 200))
        self.assertTrue(isinstance(score, float))

    def test_exif_analysis(self):
        res = ExifForensics.analyze_metadata(self.test_img_path)
        self.assertIn("exif_found", res)
        self.assertFalse(res["exif_found"]) # Dummy cv2 image won't have EXIF headers

    def test_copy_move_detection(self):
        # Image self-matching on ORB
        out_img, detected = detect_copy_move(self.test_img_path)
        self.assertIsNotNone(out_img)
        self.assertFalse(detected) # Flat dummy canvas has no copy-move patterns

    def test_double_jpeg_detection(self):
        detected = detect_double_jpeg(self.test_img_path)
        self.assertIsInstance(detected, bool)

if __name__ == "__main__":
    unittest.main()
