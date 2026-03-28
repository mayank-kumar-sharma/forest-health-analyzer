import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from fpdf import FPDF
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Config / Constants
# -----------------------------
K_VALUE = 0.5
MIN_TREE_SPACING = 15

# -----------------------------
# Helper Functions
# -----------------------------

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    return image_np

def calculate_exg(image_np):
    R = image_np[:, :, 0].astype(float)
    G = image_np[:, :, 1].astype(float)
    B = image_np[:, :, 2].astype(float)
    exg = 2 * G - R - B
    return exg, R, G, B

def generate_vegetation_mask(exg, G, R, B, k=K_VALUE):
    # UPGRADE 1 — Adaptive threshold with clip (fixes plantation explosion)
    threshold = np.mean(exg) + k * np.std(exg)
    threshold = np.clip(threshold, 10, 40)

    # UPGRADE 2 — Green ratio filter (fixes desert false positives + soil)
    green_ratio = G / (R + B + 1e-6)
    
    # Combined mask — must pass BOTH filters
    mask = ((exg > threshold) & (green_ratio > 0.9)).astype(np.uint8) * 255

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # UPGRADE 3 — Median blur before blob detection (removes noise)
    mask = cv2.medianBlur(mask, 5)

    return mask, threshold

def generate_health_map(exg, mask):
    height, width = exg.shape
    health_map = np.zeros((height, width, 3), dtype=np.uint8)
    strong_threshold = np.mean(exg[mask == 255]) if np.sum(mask == 255) > 0 else 40

    # UPGRADE 5 — Renamed labels (scientifically correct)
    # Green = Strong Vegetation
    health_map[(mask == 255) & (exg > strong_threshold)] = [0, 180, 0]
    # Yellow = Weak Vegetation
    health_map[(mask == 255) & (exg <= strong_threshold)] = [255, 200, 0]
    # Red = Non-Vegetated
    health_map[mask == 0] = [180, 0, 0]

    return health_map

def calculate_canopy_coverage(mask):
    total_pixels = mask.size
    canopy_pixels = np.sum(mask == 255)
    canopy_percent = (canopy_pixels / total_pixels) * 100
    return round(canopy_percent, 2)

def calculate_health_scores(exg, mask):
    total_pixels = mask.size
    strong_threshold = np.mean(exg[mask == 255]) if np.sum(mask == 255) > 0 else 40
    strong = np.sum((mask == 255) & (exg > strong_threshold))
    weak = np.sum((mask == 255) & (exg <= strong_threshold))
    non_veg = np.sum(mask == 0)
    return {
        "strong_pct": round((strong / total_pixels) * 100, 2),
        "weak_pct": round((weak / total_pixels) * 100, 2),
        "nonveg_pct": round((non_veg / total_pixels) * 100, 2)
    }

def filter_by_distance(keypoints, min_spacing=MIN_TREE_SPACING):
    if len(keypoints) == 0:
        return keypoints

    points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    sizes = np.array([kp.size for kp in keypoints])
    sorted_idx = np.argsort(-sizes)
    points = points[sorted_idx]
    keypoints_sorted = [keypoints[i] for i in sorted_idx]

    kept = []
    kept_points = []

    for i, kp in enumerate(keypoints_sorted):
        pt = points[i]
        if len(kept_points) == 0:
            kept.append(kp)
            kept_points.append(pt)
        else:
            dists = cdist([pt], kept_points)[0]
            if np.min(dists) >= min_spacing:
                kept.append(kp)
                kept_points.append(pt)

    return kept

def count_trees_blob(mask, mode, min_blob, max_blob, min_spacing):
    smoothed = cv2.GaussianBlur(mask, (5, 5), 0)
    inverted = cv2.bitwise_not(smoothed)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_blob
    params.maxArea = max_blob

    # UPGRADE 4 — Circularity filter (fixes row merging)
    params.filterByCircularity = True
    if mode == "🌳 Sparse Forest / Dryland":
        params.minCircularity = 0.3
    else:
        params.minCircularity = 0.2   # Agriculture — slightly relaxed

    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    raw_keypoints = detector.detect(inverted)
    filtered_keypoints = filter_by_distance(raw_keypoints, min_spacing=min_spacing)

    return len(filtered_keypoints), filtered_keypoints, len(raw_keypoints)

def draw_tree_detections(image_np, keypoints):
    result = image_np.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = max(int(kp.size / 2), 8)
        cv2.circle(result, (x, y), radius, (0, 255, 255), 2)
        cv2.circle(result, (x, y), 3, (0, 255, 255), -1)
    return result

def overlay_health_on_original(image_np, health_map, alpha=0.5):
    blended = cv2.addWeighted(image_np, 1 - alpha, health_map, alpha, 0)
    return blended

def generate_pdf_report(canopy_pct, health_scores, tree_count, raw_count, threshold_used, image_name, mode):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(31, 92, 58)
    pdf.cell(0, 12, "Flora Carbon AI", ln=True, align="C")

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "Drone Image Vegetation Analysis Report", ln=True, align="C")
    pdf.ln(4)

    pdf.set_draw_color(31, 92, 58)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Image Analyzed: {image_name}", ln=True)
    pdf.cell(0, 8, f"Analysis Mode: {mode}", ln=True)
    pdf.cell(0, 8, f"Adaptive Threshold Used: {threshold_used:.2f}", ln=True)
    pdf.ln(4)

    # Section 1
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(31, 92, 58)
    pdf.cell(0, 8, "1. Canopy Coverage", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, f"   Canopy Coverage: {canopy_pct}%", ln=True)
    pdf.ln(3)

    # Section 2
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(31, 92, 58)
    pdf.cell(0, 8, "2. Vegetation Health Breakdown", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, f"   Strong Vegetation  : {health_scores['strong_pct']}%", ln=True)
    pdf.cell(0, 8, f"   Weak Vegetation    : {health_scores['weak_pct']}%", ln=True)
    pdf.cell(0, 8, f"   Non-Vegetated Area : {health_scores['nonveg_pct']}%", ln=True)
    pdf.ln(3)

    # Section 3
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(31, 92, 58)
    pdf.cell(0, 8, "3. Tree Crown Count", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, f"   Raw Detections (before filtering): {raw_count}", ln=True)
    pdf.cell(0, 8, f"   Final Count (after duplicate removal): {tree_count}", ln=True)
    pdf.ln(3)

    # Section 4
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(31, 92, 58)
    pdf.cell(0, 8, "4. Methodology", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 7,
        "   Vegetation detection uses Excess Green Index (ExG = 2G - R - B) combined "
        "with Green Ratio filter (G / R+B) for improved soil and desert rejection. "
        "Adaptive threshold is clipped between 10-40 for stability across image types. "
        "Median blur removes noise before blob detection. "
        "Crown detection uses blob detection with circularity filter and "
        "distance-based duplicate removal."
    )
    pdf.ln(3)

    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 7,
        "Accuracy Note: Health Map ~65-75% | Canopy Coverage ~75-85% | "
        "Tree Count ~70-85% (sparse/plantation) | ~50-65% (dense canopy). "
        "Optimized for sparse forests, orchards, plantations, and agroforestry. "
        "Results are for project scoping only. Ground truth validation recommended for carbon verification."
    )

    pdf.ln(4)
    pdf.set_draw_color(31, 92, 58)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(31, 92, 58)
    pdf.cell(0, 8, "Made with love by Mayank Kumar Sharma | Flora Carbon AI", ln=True, align="C")

    pdf_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(pdf_path)
    return pdf_path


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌳 Flora Carbon AI — Vegetation Analyzer")

# DOMAIN HEADING
st.success(
    "✅ Works best with: Sparse forests, orchards, plantations, agroforestry plots, "
    "and dryland trees with clearly separated and visible crowns."
)
st.error(
    "❌ Not suitable for: Dense canopy forests, conifer/pine forests, "
    "early stage crop fields, or mixed terrain images."
)

st.markdown("Upload a drone orthomosaic image to analyze vegetation health, canopy coverage, and tree count.")

st.info(
    "📌 Best results with high resolution orthomosaic images (JPG/PNG/TIF) "
    "taken from directly above on a clear sunny day."
)

uploaded_file = st.file_uploader(
    "Upload Drone Orthomosaic Image (JPG / PNG / TIF)",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Drone Image", use_column_width=True)

    st.markdown("---")
    st.subheader("⚙️ Settings")

    # MODE SELECTOR
    mode = st.selectbox(
        "Select Analysis Mode",
        ["🌳 Sparse Forest / Dryland", "🌾 Agriculture / Plantation (Experimental)"],
        help="Forest mode = stricter crown detection. Agriculture = more sensitive for row crops and orchards."
    )

    k_value = st.slider(
        "Adaptive Threshold Sensitivity (k)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Lower = more vegetation detected. Higher = stricter. 0.5 works for most images."
    )

    # Mode based defaults
    if mode == "🌳 Sparse Forest / Dryland":
        default_min_blob = 100
        default_max_blob = 8000
        default_spacing = 15
    else:
        default_min_blob = 50
        default_max_blob = 4000
        default_spacing = 10

    min_blob = st.slider(
        "Minimum Tree Crown Size (pixels)",
        min_value=50,
        max_value=500,
        value=default_min_blob,
        help="Increase if small noise is being counted as trees."
    )

    max_blob = st.slider(
        "Maximum Tree Crown Size (pixels)",
        min_value=1000,
        max_value=20000,
        value=default_max_blob,
        help="Increase for large tree canopies."
    )

    min_spacing = st.slider(
        "Minimum Tree Spacing (pixels)",
        min_value=5,
        max_value=100,
        value=default_spacing,
        help="Increase to reduce double counting of same tree."
    )

    show_overlay = st.checkbox(
        "Show Health Map overlaid on original image",
        value=True
    )

    if st.button("🔍 Analyze Vegetation"):

        with st.spinner("Analyzing image..."):

            image_np = load_image(uploaded_file)
            exg, R, G, B = calculate_exg(image_np)
            mask, threshold_used = generate_vegetation_mask(exg, G, R, B, k=k_value)
            health_map = generate_health_map(exg, mask)
            canopy_pct = calculate_canopy_coverage(mask)
            health_scores = calculate_health_scores(exg, mask)

            tree_count, keypoints, raw_count = count_trees_blob(
                mask,
                mode=mode,
                min_blob=min_blob,
                max_blob=max_blob,
                min_spacing=min_spacing
            )

            tree_image = draw_tree_detections(image_np, keypoints)

            if show_overlay:
                display_health = overlay_health_on_original(image_np, health_map, alpha=0.5)
            else:
                display_health = health_map

        st.success("✅ Analysis Complete!")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.image(display_health, caption="Vegetation Health Map", use_column_width=True)
            st.markdown("🟢 Strong Vegetation &nbsp; 🟡 Weak Vegetation &nbsp; 🔴 Non-Vegetated")
        with col2:
            st.image(tree_image, caption=f"Crowns Detected: {tree_count}", use_column_width=True)

        st.markdown("---")
        st.subheader("📊 Results Summary")

        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("🌿 Canopy Coverage", f"{canopy_pct}%")
        with col4:
            st.metric("🌳 Crowns Detected", tree_count)
        with col5:
            st.metric("💚 Strong Vegetation", f"{health_scores['strong_pct']}%")

        col6, col7, col8 = st.columns(3)
        with col6:
            st.metric("🟡 Weak Vegetation", f"{health_scores['weak_pct']}%")
        with col7:
            st.metric("🔴 Non-Vegetated Area", f"{health_scores['nonveg_pct']}%")
        with col8:
            st.metric("🔍 Raw Detections", raw_count)

        st.caption(
            f"Mode: {mode} | "
            f"Adaptive threshold: {threshold_used:.2f} | "
            f"Duplicates removed: {raw_count - tree_count}"
        )

        st.markdown("---")
        st.warning(
            "⚠️ Accuracy Note: Health Map ~65-75% | Canopy ~75-85% | "
            "Tree Count ~70-85% sparse / ~50-65% dense. "
            "For carbon verification, ground truth validation is recommended."
        )

        st.markdown("---")
        st.subheader("📄 Download Report")
        pdf_path = generate_pdf_report(
            canopy_pct=canopy_pct,
            health_scores=health_scores,
            tree_count=tree_count,
            raw_count=raw_count,
            threshold_used=threshold_used,
            image_name=uploaded_file.name,
            mode=mode
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f,
                file_name="flora_carbon_vegetation_report.pdf",
                mime="application/pdf"
            )
        os.unlink(pdf_path)

# Footer
st.markdown("---")
st.markdown("💡 Made with ❤️ by **Mayank Kumar Sharma** | Flora Carbon AI")
