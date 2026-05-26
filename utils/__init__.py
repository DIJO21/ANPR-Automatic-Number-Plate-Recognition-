# Utilities initialization
from .gpu import get_device, clear_gpu_memory, get_gpu_info, print_gpu_summary
from .tracking import LicensePlateTracker
from .datasets import setup_kaggle_credentials, download_forensic_datasets, clean_corrupted_images, generate_mock_anpr_dataset
from .report_generator import ForensicReportGenerator
