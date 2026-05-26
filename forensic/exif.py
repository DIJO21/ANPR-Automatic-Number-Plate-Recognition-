from PIL import Image
from PIL.ExifTags import TAGS
import logging

logger = logging.getLogger("ForensicANPR.EXIF")

class ExifForensics:
    """Analyzes image EXIF headers for signatures of digital alteration."""

    SUSPICIOUS_SOFTWARE = [
        "PHOTOSHOP", "GIMP", "CANVA", "PICSART", "LIGHTROOM", 
        "PIXLR", "IMAGE MAGICK", "SNAPSEED", "ADOBE"
    ]

    @classmethod
    def analyze_metadata(cls, image_path: str) -> dict:
        """Extracts and inspects EXIF metadata for forgery footprints."""
        results = {
            "exif_found": False,
            "exif_tampered": False,
            "software": None,
            "creation_date": None,
            "camera_make": None,
            "camera_model": None,
            "details": "No EXIF metadata found."
        }

        try:
            with Image.open(image_path) as img:
                exif_data = img.getexif()
                if not exif_data:
                    results["details"] = "No EXIF header found. (Standard behavior for web downloads or screenshots, but suspicious for raw camera evidence)."
                    return results

                results["exif_found"] = True
                extracted_tags = {}
                
                # Parse standard EXIF tags
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    extracted_tags[tag_name] = value

                # Check for Software editing tags
                software = extracted_tags.get("Software", "")
                if software:
                    results["software"] = str(software)
                    software_upper = str(software).upper()
                    for term in cls.SUSPICIOUS_SOFTWARE:
                        if term in software_upper:
                            results["exif_tampered"] = True
                            results["details"] = f"Image edited using software signature: {software}"
                            break
                
                results["creation_date"] = extracted_tags.get("DateTime", extracted_tags.get("DateTimeOriginal", None))
                results["camera_make"] = extracted_tags.get("Make", None)
                results["camera_model"] = extracted_tags.get("Model", None)

                # Flag missing device details in present EXIF as potential strip-forgery
                if not results["exif_tampered"]:
                    if not results["camera_make"] and not results["camera_model"] and results["software"]:
                        results["exif_tampered"] = True
                        results["details"] = f"EXIF headers contain software details ({software}) but missing device capture details. Likely edited."
                    else:
                        results["details"] = "EXIF metadata is present and contains no obvious software forgery signatures."

        except Exception as e:
            logger.error(f"Error parsing EXIF: {str(e)}")
            results["details"] = f"Failed to parse EXIF structure: {str(e)}"

        return results
