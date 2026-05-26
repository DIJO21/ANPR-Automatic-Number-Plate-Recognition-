import unittest
from decoder.indian_plates import IndianPlateParser
from decoder.probabilistic import ProbabilisticDecoder, WatchlistMatcher

class TestIndianPlates(unittest.TestCase):
    """Unit tests for RTO standard pattern validations."""

    def test_standard_plate(self):
        res = IndianPlateParser.validate_and_parse("MH12GP1234")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["plate_type"], "Standard")
        self.assertEqual(res["state_code"], "MH")
        self.assertEqual(res["number"], 1234)

        # Test Delhi style zone letters
        res_delhi = IndianPlateParser.validate_and_parse("DL3CAY1111")
        self.assertTrue(res_delhi["is_valid"])
        self.assertEqual(res_delhi["rto_zone"], "3C")

    def test_bharat_series(self):
        res = IndianPlateParser.validate_and_parse("21BH1234AA")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["plate_type"], "Bharat Series (BH)")
        self.assertEqual(res["year_of_registration"], "2021")

    def test_military_plates(self):
        res = IndianPlateParser.validate_and_parse("↑15D123456K")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["plate_type"], "Military")
        self.assertEqual(res["base_code"], "D")

    def test_diplomatic_plates(self):
        res = IndianPlateParser.validate_and_parse("11CD21")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["plate_type"], "Diplomatic / Consular")
        self.assertEqual(res["embassy_code"], 11)

    def test_invalid_plates(self):
        res = IndianPlateParser.validate_and_parse("ABC123XYZ")
        self.assertFalse(res["is_valid"])

class TestProbabilisticDecoder(unittest.TestCase):
    """Unit tests for position-aware OCR error correction."""

    def test_confusion_corrections(self):
        # Index 0,1 are State code -> '0' should correct to 'O'
        raw = "0H12GP1234" # MH12... typo
        corrected = ProbabilisticDecoder.correct_confusions(raw)
        self.assertEqual(corrected[0], "O") # Corrected 0 to O

        # Last characters are numbers -> 'O' should correct to '0'
        raw_num = "MH12GP123O"
        corrected_num = ProbabilisticDecoder.correct_confusions(raw_num)
        self.assertEqual(corrected_num[-1], "0")

        # Test that shorter plates do not get corrupted (e.g. MH12GP44 -> G is not converted to 6)
        raw_short = "MH12GP44"
        corrected_short = ProbabilisticDecoder.correct_confusions(raw_short)
        self.assertEqual(corrected_short, "MH12GP44")

    def test_watchlist_matching(self):
        watchlist = ["MH12GP1234", "DL3CAY1111"]
        matcher = WatchlistMatcher(watchlist)
        
        # Exact match
        matches = matcher.search("MH12GP1234")
        self.assertEqual(matches[0][1], 1.0)
        
        # Levenshtein distance 1 match
        matches_sim = matcher.search("MH12GP123O") # ending with 'O'
        self.assertGreater(len(matches_sim), 0)
        self.assertTrue(matches_sim[0][1] >= 0.8)

    def test_generate_alternatives(self):
        alternatives = ProbabilisticDecoder.generate_alternatives("KLO8AH1509", max_changes=1)
        # Should generate KL08AH1509 by correcting O to 0 (changes = 1)
        self.assertIn("KL08AH1509", alternatives)
        # Should contain original text
        self.assertIn("KLO8AH1509", alternatives)

if __name__ == "__main__":
    unittest.main()
