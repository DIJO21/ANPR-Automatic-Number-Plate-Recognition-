import os
import cv2
import zipfile
import json
import logging
import datetime
import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
from streamlit_cropper import st_cropper

from inference import ForensicANPRPipeline
from utils.gpu import get_gpu_info
from utils.report_generator import ForensicReportGenerator
from forensic import compute_ela

# Initialise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForensicANPR.UI")

# Page Configuration for Premium Aesthetic
st.set_page_config(
    page_title="Forensic ANPR Terminal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism CSS styling for Wow-factor
st.markdown("""
<style>
    /* Dark Deep-Navy theme */
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0B1329 100%);
        color: #F8FAFC;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Neon Glow Borders and Headers */
    .forensic-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #F43F5E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
        filter: drop-shadow(0px 4px 12px rgba(99, 102, 241, 0.25));
    }
    
    .forensic-subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-top: -15px;
        margin-bottom: 30px;
    }

    /* Glassmorphic panels */
    div[data-testid="stMetricValue"] {
        color: #38BDF8;
        font-size: 2rem !important;
        font-weight: 700;
    }
    
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    .status-alert-tampered {
        background: rgba(220, 38, 38, 0.15);
        border: 1px solid #EF4444;
        border-radius: 8px;
        padding: 15px;
        color: #FCA5A5;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .status-alert-clean {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        border-radius: 8px;
        padding: 15px;
        color: #A7F3D0;
        font-weight: 600;
        margin-bottom: 15px;
    }

    /* Glow buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# Setup workspace directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Case History Management functions
def load_history():
    history_file = OUTPUT_DIR / "history.json"
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_to_history(case_id, filename, plates, forensics):
    history = load_history()
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_id": case_id,
        "filename": filename,
        "plates_detected": [p["corrected_ocr"] for p in plates],
        "is_manipulated": forensics.get("is_manipulated", False),
        "ela_score": float(forensics.get("ela_score", 0.0)),
        "watchlist_status": plates[0]["watchlist_status"] if len(plates) > 0 else "CLEAN"
    }
    history.append(entry)
    # Save last 50 investigations
    history = history[-50:]
    try:
        with open(OUTPUT_DIR / "history.json", 'w') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to write history: {e}")

# Manage Watchlist session state
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["DL3CAY1111", "MH12GP1234", "KA51MB8899"]

# Initialize Pipeline
@st.cache_resource
def get_pipeline():
    weights = "weights/best.pt" if os.path.exists("weights/best.pt") else "yolov8n.pt"
    return ForensicANPRPipeline(
        yolo_weights=weights, 
        ocr_weights=None, 
        watchlist=st.session_state.watchlist
    )

pipeline = get_pipeline()

# Title banner
st.markdown("<h1 class='forensic-title'>🛡️ CYBER FORENSICS ANPR TERMINAL</h1>", unsafe_allow_html=True)
st.markdown("<p class='forensic-subtitle'>National Vehicle Verification & Image Tampering Analysis Engine</p>", unsafe_allow_html=True)

# SIDEBAR - Investigative Settings
st.sidebar.markdown("### 🖥️ Investigator Credentials")
case_id = st.sidebar.text_input("Case ID / Docket Number", "FED-2026-9031")
investigator = st.sidebar.text_input("Lead Investigator", "Agent Deepak Kumar")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Operation Mode")
mode = st.sidebar.radio(
    "Choose Input Stream:",
    ["🖥️ Image Analysis", "🎞️ Video Frame Analysis", "📋 Case History Log", "📁 Watchlist Manager"]
)

# Hardware Diagnostics Panel
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔋 System Diagnostics")
gpu_stats = get_gpu_info()
if gpu_stats["cuda_available"]:
    st.sidebar.success(f"GPU Accelerator Active: {gpu_stats['current_device_name']}")
    st.sidebar.metric("VRAM Reserved", f"{gpu_stats['reserved_mb']:.1f} MB")
else:
    st.sidebar.warning("CPU Fallback Active (No CUDA device)")

# ----------------- WATCHLIST MANAGER MODE -----------------
if mode == "📁 Watchlist Manager":
    st.markdown("<div class='glass-card'><h3>📋 Watchlist Database Control</h3>", unsafe_allow_html=True)
    st.write("Active monitored license plates:")
    st.code(", ".join(st.session_state.watchlist))
    
    new_plate = st.text_input("Add suspicious/stolen plate to Watchlist:")
    if st.button("Register Target"):
        if new_plate:
            plate_std = new_plate.replace(" ", "").upper()
            if plate_std not in st.session_state.watchlist:
                st.session_state.watchlist.append(plate_std)
                st.success(f"Target registered: {plate_std}")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.warning("Plate is already registered on Watchlist.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- CASE HISTORY LOG MODE -----------------
elif mode == "📋 Case History Log":
    st.markdown("<div class='glass-card'><h3>📜 Investigative Case History</h3>", unsafe_allow_html=True)
    history = load_history()
    
    if len(history) == 0:
        st.info("No past case analyses recorded in database.")
    else:
        st.write(f"Total entries recorded: **{len(history)}**")
        # Format table representation
        table_data = []
        for idx, entry in enumerate(reversed(history)):
            table_data.append({
                "Index": len(history) - idx,
                "Timestamp": entry["timestamp"],
                "Case ID": entry["case_id"],
                "File Target": entry["filename"],
                "Plates Read": ", ".join(entry["plates_detected"]) if entry["plates_detected"] else "None",
                "Integrity Status": "TAMPERED ⚠️" if entry["is_manipulated"] else "VERIFIED 🛡️",
                "Watchlist Status": entry["watchlist_status"]
            })
        st.table(table_data)
        
        if st.button("Clear History Log"):
            try:
                (OUTPUT_DIR / "history.json").unlink(missing_ok=True)
                st.success("History log database cleared successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear database: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- VIDEO FRAME CROPPING MODE -----------------
elif mode == "🎞️ Video Frame Analysis":
    st.markdown("<div class='glass-card'><h3>🎞️ Video Timeline Frame Extractor</h3>", unsafe_allow_html=True)
    video_file = st.file_uploader("Upload video file (MP4, AVI)", type=["mp4", "avi"])
    
    if video_file:
        temp_input_path = UPLOAD_DIR / f"video_{video_file.name}"
        with open(temp_input_path, "wb") as f:
            f.write(video_file.getbuffer())

        # Retrieve video frame characteristics
        cap = cv2.VideoCapture(str(temp_input_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / fps
        
        st.write(f"Video loaded successfully: **{total_frames} frames** (Duration: {duration_sec:.2f} seconds)")
        
        # Frame selection timeline slider
        frame_idx = st.slider("Select Frame to Extract (Timeline):", 0, total_frames - 1, 0)
        
        # Read the selected frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR frame to PIL RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            col_crop, col_preview = st.columns([6, 4])
            
            with col_crop:
                st.markdown("#### Crop Plate Target from Selected Frame:")
                # Crop Tool on extracted frame
                cropped_img = st_cropper(frame_pil, realtime_update=True, box_color='#E11D48', aspect_ratio=None)
            
            with col_preview:
                st.markdown("#### Crop Preview:")
                st.image(cropped_img, use_container_width=True)
                
                # Execution Trigger
                analyze_click = st.button("🚀 Analyze Cropped Video Region")
                
            if analyze_click:
                temp_crop_path = UPLOAD_DIR / f"extracted_frame_{video_file.name}.jpg"
                cropped_img.save(temp_crop_path)
                
                with st.spinner("Processing extracted video region..."):
                    results = pipeline.process_image(str(temp_crop_path))
                
                forensics = results["forensics"]
                is_manipulated = forensics["is_manipulated"]
                
                # Save visual overlays
                annotated_path = OUTPUT_DIR / f"annotated_frame_{video_file.name}.jpg"
                cv2.imwrite(str(annotated_path), results["annotated_image"])
                
                ela_map, _ = compute_ela(str(temp_crop_path))
                ela_path = OUTPUT_DIR / f"ela_frame_{video_file.name}.jpg"
                cv2.imwrite(str(ela_path), cv2.cvtColor(ela_map, cv2.COLOR_RGB2BGR))
                
                save_to_history(case_id, f"{video_file.name} (Frame {frame_idx})", results["plates_detected"], forensics)
                
                # Display Results Panel
                col_res1, col_res2 = st.columns([6, 4])
                with col_res1:
                    st.image(str(annotated_path), use_container_width=True)
                with col_res2:
                    if is_manipulated:
                        st.markdown("<div class='status-alert-tampered'>⚠️ IMAGE INTEGRITY BREACHED: TAMPERING DETECTED</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='status-alert-clean'>🛡️ IMAGE INTEGRITY VERIFIED: PASS</div>", unsafe_allow_html=True)
                    st.metric("Error Level Analysis (ELA)", f"{forensics['ela_score']:.2f}")
                    st.write(f"**EXIF Verification:** {forensics['exif_details']}")
                    st.write(f"**Copy-Move Duplicate Matching:** {'TAMPERED' if forensics['copy_move_detected'] else 'NOT DETECTED'}")
                
                st.markdown("---")
                if len(results["plates_detected"]) == 0:
                    st.warning("No license plates detected in crop region. Fallback OCR engine initiated.")
                else:
                    for plate in results["plates_detected"]:
                        st.markdown(f"### 🚗 Plate Detected (ID: {plate['id']})")
                        col_v_a, col_v_b = st.columns(2)
                        with col_v_a:
                            st.write(f"**Decoded Text:** `{plate['corrected_ocr']}` (Confidence: {plate['ocr_confidence']:.2%})")
                            st.write(f"**Raw OCR Output:** `{plate['raw_ocr']}`")
                        with col_v_b:
                            st.write(f"**RTO Jurisdiction:** {plate['rto_info']['details']}")
                            st.write(f"**Format Type:** {plate['rto_info']['plate_type']}")
                        
                        # Display variations
                        if "variations" in plate and plate["variations"]:
                            st.markdown("##### 🔍 Alternate OCR Candidate Combinations:")
                            var_table = []
                            for idx_v, var in enumerate(plate["variations"]):
                                if idx_v >= 10:
                                    break
                                var_table.append({
                                    "No.": idx_v + 1,
                                    "Candidate Plate": var["plate"],
                                    "Format Type": var["rto_info"]["plate_type"],
                                    "Format Valid": "✅ Yes" if var["is_valid"] else "❌ No",
                                    "Watchlist": "🚨 STOLEN/WANTED" if var["watchlist_status"] == "STOLEN / WANTED" else "🟢 Clean",
                                    "RTO Jurisdiction Description": var["rto_info"]["details"]
                                })
                            st.table(var_table)
                        
                        st.markdown("---")
                
                if temp_crop_path.exists():
                    temp_crop_path.unlink()
                    
        # Cleanup original video file
        if temp_input_path.exists():
            temp_input_path.unlink()
            
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- IMAGE ANALYSIS & CROPPING MODE -----------------
else:
    st.markdown("<div class='glass-card'><h3>📷 Visual Evidence Uploader</h3>", unsafe_allow_html=True)
    image_file = st.file_uploader("Upload evidentiary image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if image_file:
        temp_input_path = UPLOAD_DIR / f"temp_{image_file.name}"
        with open(temp_input_path, "wb") as f:
            f.write(image_file.getbuffer())

        original_image = Image.open(temp_input_path)
        
        # UI controls for selecting analysis type
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col_select1, col_select2 = st.columns([7, 3])
        
        with col_select1:
            st.markdown("#### Crop Target Plate Region:")
            # Render interactive st_cropper directly on screen
            cropped_img = st_cropper(original_image, realtime_update=True, box_color='#E11D48', aspect_ratio=None)
            
        with col_select2:
            st.markdown("#### Crop Preview:")
            st.image(cropped_img, use_container_width=True)
            
            # Submit action button
            submit_click = st.button("🚀 Submit Crop for Forensic Analysis")
            
        st.markdown("</div>", unsafe_allow_html=True)

        if submit_click:
            # Save the cropped region to a temporary file
            temp_crop_path = UPLOAD_DIR / f"crop_{image_file.name}"
            cropped_img.save(temp_crop_path)
            
            with st.spinner("Analyzing cropped region..."):
                results = pipeline.process_image(str(temp_crop_path))

            forensics = results["forensics"]
            is_manipulated = forensics["is_manipulated"]

            # Save annotated image and ELA maps
            annotated_path = OUTPUT_DIR / f"annotated_{image_file.name}"
            cv2.imwrite(str(annotated_path), results["annotated_image"])

            ela_map, _ = compute_ela(str(temp_crop_path))
            ela_path = OUTPUT_DIR / f"ela_{image_file.name}"
            cv2.imwrite(str(ela_path), cv2.cvtColor(ela_map, cv2.COLOR_RGB2BGR))

            # Save to case history
            save_to_history(case_id, image_file.name, results["plates_detected"], forensics)

            # Display outputs
            col1, col2 = st.columns([6, 4])
            with col1:
                st.markdown("<div class='glass-card'><h4>Visual Plate Bounding Boxes</h4>", unsafe_allow_html=True)
                st.image(str(annotated_path), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='glass-card'><h4>Evidentiary Integrity Diagnostics</h4>", unsafe_allow_html=True)
                if is_manipulated:
                    st.markdown("<div class='status-alert-tampered'>⚠️ IMAGE INTEGRITY BREACHED: TAMPERING DETECTED</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='status-alert-clean'>🛡️ IMAGE INTEGRITY VERIFIED: PASS</div>", unsafe_allow_html=True)
                
                st.metric("Error Level Analysis (ELA)", f"{forensics['ela_score']:.2f}")
                st.write(f"**EXIF Verification:** {forensics['exif_details']}")
                st.write(f"**Double JPEG Compression:** {'ANOMALOUS / PRESENT' if forensics['double_jpeg_detected'] else 'NOT DETECTED'}")
                st.write(f"**Copy-Move Duplicate Matching:** {'TAMPERED / DETECTED' if forensics['copy_move_detected'] else 'NOT DETECTED'}")
                st.markdown("</div>", unsafe_allow_html=True)

            # Plates Details Panel
            st.markdown("<div class='glass-card'><h3>🚗 Decoded Vehicle Information</h3>", unsafe_allow_html=True)
            if len(results["plates_detected"]) == 0:
                st.warning("No license plates detected in crop region. Fallback OCR engine initiated.")
            else:
                for plate in results["plates_detected"]:
                    rto = plate["rto_info"]
                    watchlist_status = plate["watchlist_status"]
                    
                    c_status_class = "status-alert-tampered" if watchlist_status == "STOLEN / WANTED" else "status-alert-clean"
                    st.markdown(f"<div class='{c_status_class}'>Watchlist Code: {watchlist_status}</div>", unsafe_allow_html=True)

                    col_crop_box, col_a, col_b = st.columns([3, 5, 5])
                    with col_crop_box:
                        st.markdown("**Visual Plate Crop:**")
                        x1, y1, x2, y2 = plate["box"]
                        try:
                            with Image.open(temp_crop_path) as orig_img:
                                crop_img = orig_img.crop((x1, y1, x2, y2))
                                st.image(crop_img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error cropping plate: {e}")

                    with col_a:
                        st.write(f"**Raw OCR Output:** `{plate['raw_ocr']}`")
                        st.write(f"**Corrected Standard OCR:** `{plate['corrected_ocr']}`")
                        st.write(f"**Confidence:** `{plate['ocr_confidence']:.2%}`")
                    
                    with col_b:
                        st.write(f"**Registration Type:** {rto['plate_type']}")
                        st.write(f"**Assigned Code:** `{rto['formatted']}`")
                        st.write(f"**RTO Jurisdiction Description:** {rto['details']}")

                    # Display variations
                    if "variations" in plate and plate["variations"]:
                        st.markdown("##### 🔍 Alternate OCR Candidate Combinations:")
                        var_table = []
                        for idx_v, var in enumerate(plate["variations"]):
                            if idx_v >= 10:
                                break
                            var_table.append({
                                "No.": idx_v + 1,
                                "Candidate Plate": var["plate"],
                                "Format Type": var["rto_info"]["plate_type"],
                                "Format Valid": "✅ Yes" if var["is_valid"] else "❌ No",
                                "Watchlist": "🚨 STOLEN/WANTED" if var["watchlist_status"] == "STOLEN / WANTED" else "🟢 Clean",
                                "RTO Jurisdiction Description": var["rto_info"]["details"]
                            })
                        st.table(var_table)

                    st.markdown("---")
            st.markdown("</div>", unsafe_allow_html=True)

            # Export Controls
            st.markdown("<div class='glass-card'><h3>📥 Forensic Report Compiler</h3>", unsafe_allow_html=True)
            
            doc_plate = results["plates_detected"][0]["corrected_ocr"] if len(results["plates_detected"]) > 0 else "UNKNOWN"
            rto_meta = results["plates_detected"][0]["rto_info"] if len(results["plates_detected"]) > 0 else {}
            rto_meta["watchlist_status"] = results["plates_detected"][0]["watchlist_status"] if len(results["plates_detected"]) > 0 else "CLEAN"

            report_filename = f"Report_{case_id}_{image_file.name}.pdf"
            report_path = OUTPUT_DIR / report_filename

            report_gen = ForensicReportGenerator(str(report_path))
            report_gen.generate(
                case_id=case_id,
                investigator=investigator,
                plate_text=doc_plate,
                original_img_path=str(temp_crop_path),
                ela_img_path=str(ela_path),
                metadata=rto_meta,
                forensics=forensics
            )

            zip_filename = f"Evidence_{case_id}_{image_file.name}.zip"
            zip_path = OUTPUT_DIR / zip_filename
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(temp_crop_path, arcname=image_file.name)
                zipf.write(annotated_path, arcname=f"annotated_{image_file.name}")
                zipf.write(ela_path, arcname=f"ela_{image_file.name}")
                # Write JSON details
                json_details = {
                    "case_id": case_id,
                    "investigator": investigator,
                    "forensic_diagnostics": forensics,
                    "plates": results["plates_detected"]
                }
                json_path = OUTPUT_DIR / f"metadata_{case_id}.json"
                with open(json_path, 'w') as jf:
                    json.dump(json_details, jf, default=str, indent=4)
                zipf.write(json_path, arcname="forensic_metadata.json")
                json_path.unlink()

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Certified PDF Forensic Report",
                        data=f,
                        file_name=report_filename,
                        mime="application/pdf"
                    )
            with col_dl2:
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Complete Evidence Zip Archive",
                        data=f,
                        file_name=zip_filename,
                        mime="application/zip"
                    )
            st.markdown("</div>", unsafe_allow_html=True)

            if temp_crop_path.exists():
                temp_crop_path.unlink()

        # Clean up files in uploads
        if temp_input_path.exists():
            temp_input_path.unlink()
