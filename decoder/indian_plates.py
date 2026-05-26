import re
from typing import Dict, Optional, Tuple

# RTO State/UT Codes Lookup Table
STATE_CODES: Dict[str, str] = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CH": "Chandigarh",
    "CG": "Chhattisgarh",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "DD": "Daman and Diu (Legacy)",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha (Legacy)",
    "PY": "Puducherry",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand (Legacy)",
    "WB": "West Bengal"
}

class IndianPlateParser:
    """Validator and information extractor for Indian vehicle registration plates."""

    # Regex definitions
    # 1. Standard: e.g., MH 12 GP 1234, DL 3C AY 1111, HR 26 DK 8332
    # State (2 chars), RTO Zone (2 digits or 1 digit + 1 letter), Series (1-2 letters), Number (1-4 digits)
    STANDARD_REGEX = re.compile(r"^([A-Z]{2})\s*([0-9]{1,2}[A-Z]?)\s*([A-Z]{1,2})\s*([0-9]{1,4})$")

    # 2. Bharat Series: e.g., 21 BH 1234 AA
    # Year (2 digits), BH (literal), Number (4 digits), Series (2 letters)
    BHARAT_REGEX = re.compile(r"^([0-9]{2})\s*(BH)\s*([0-9]{4})\s*([A-Z]{2})$")

    # 3. Military: e.g., ↑15D123456K or ^15D123456K
    # Broad arrow (↑ or ^), Year (2 digits), Base Code (1 letter), Number (6 digits), Check Letter (1 letter)
    MILITARY_REGEX = re.compile(r"^[↑^]\s*([0-9]{2})\s*([A-Z])\s*([0-9]{6})\s*([A-Z])$")

    # 4. Diplomatic/Consular: e.g., 11 CD 21, 77 UN 100
    # Country/Embassy Code (1-3 digits), CD/CC/UN, Vehicle Number (1-4 digits)
    DIPLOMATIC_REGEX = re.compile(r"^([0-9]{1,3})\s*(CD|CC|UN)\s*([0-9]{1,4})$")

    @classmethod
    def preprocess_plate_text(cls, text: str) -> str:
        """Cleans and standardizes raw OCR outputs (strips spaces, symbols, uppercase conversion)."""
        if not text:
            return ""
        # Convert to upper and strip leading/trailing spaces
        cleaned = text.upper().strip()
        # Remove common OCR trailing/leading garbage, keeping letters, digits, and the military arrow symbol
        cleaned = re.sub(r"[^A-Z0-9↑^]", "", cleaned)
        return cleaned

    @classmethod
    def validate_and_parse(cls, raw_text: str) -> dict:
        """Validates and parses the plate against standard Indian vehicle plate registration templates."""
        cleaned = cls.preprocess_plate_text(raw_text)
        
        # 1. Match Standard Plates
        std_match = cls.STANDARD_REGEX.match(cleaned)
        if std_match:
            state_code, rto_zone, series, number = std_match.groups()
            state_name = STATE_CODES.get(state_code)
            if state_name:
                return {
                    "is_valid": True,
                    "plate_type": "Standard",
                    "state_code": state_code,
                    "state_name": state_name,
                    "rto_zone": rto_zone,
                    "series": series,
                    "number": int(number),
                    "formatted": f"{state_code} {rto_zone} {series} {int(number):04d}",
                    "details": f"Registered in {state_name} (RTO Zone: {rto_zone}, Series: {series})"
                }

        # 2. Match Bharat Series
        bh_match = cls.BHARAT_REGEX.match(cleaned)
        if bh_match:
            year, bh, number, series = bh_match.groups()
            return {
                "is_valid": True,
                "plate_type": "Bharat Series (BH)",
                "year_of_registration": f"20{year}",
                "number": int(number),
                "series": series,
                "formatted": f"{year} BH {int(number):04d} {series}",
                "details": f"Bharat Series registered in 20{year} (Series: {series})"
            }

        # 3. Match Military Series
        mil_match = cls.MILITARY_REGEX.match(cleaned)
        if mil_match:
            year, base_code, number, check_letter = mil_match.groups()
            return {
                "is_valid": True,
                "plate_type": "Military",
                "year_of_registration": f"20{year}" if int(year) < 50 else f"19{year}",
                "base_code": base_code,
                "number": int(number),
                "check_letter": check_letter,
                "formatted": f"↑ {year} {base_code} {int(number):06d} {check_letter}",
                "details": f"Indian Armed Forces vehicle registered in {year} (Base: {base_code})"
            }

        # 4. Match Diplomatic Series
        dip_match = cls.DIPLOMATIC_REGEX.match(cleaned)
        if dip_match:
            embassy_code, body, number = dip_match.groups()
            body_map = {"CD": "Diplomatic Corps", "CC": "Consular Corps", "UN": "United Nations"}
            return {
                "is_valid": True,
                "plate_type": "Diplomatic / Consular",
                "embassy_code": int(embassy_code),
                "diplomatic_body": body_map.get(body, body),
                "number": int(number),
                "formatted": f"{embassy_code} {body} {number}",
                "details": f"Embassy/Consular vehicle of Country code {embassy_code} ({body_map.get(body)})"
            }

        # Check for temporary formatting heuristically
        if "TEMP" in cleaned or "TMP" in cleaned:
            return {
                "is_valid": False,
                "plate_type": "Temporary (Non-standard)",
                "formatted": cleaned,
                "details": "Temporary/Trade plate detected. Failed standard formatting checks."
            }

        return {
            "is_valid": False,
            "plate_type": "Invalid / Foreign / Unrecognized",
            "formatted": raw_text,
            "details": "Does not conform to any standard Indian registration plates template."
        }
