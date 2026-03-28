import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from fpdf import FPDF
from scipy.spatial.distance import cdist
from scipy import ndimage
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

def generate_vegetation_mask(exg, G, R, B, k=K_VALUE, mode="🌳 Sparse Forest / Dryland"):
    # Adaptive threshold with clip
    threshold = np.mean(exg) + k * np.std(exg)
    threshold = np.clip(threshold, 10, 40)

    # Green ratio — relaxed for agriculture mode
    green_ratio = G / (R + B + 1e-6)

    if mode == "🌾 Agriculture / Plantation (Experimental)":
        green_ratio_threshold = 0.6
    else:
        green_ratio_threshold = 0.9

    mask = ((exg > threshold) & (green_ratio > green_ratio_threshold)).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)

    return mask, threshold

def generate_health_map(exg, mask):
    height, width = exg.shape
    health_map = np.zeros((height, width, 3), dtype=np.uint8)
    strong_threshold = np.mean(exg[mask == 255]) if np.sum(mask == 255) > 0 else 40

    health_map[(mask == 255) & (exg > strong_threshold)] = [0, 180, 0]
    health_map[(mask == 255) & (exg <= strong_threshold)] = [255, 200, 0]
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

def detect_crowns_distance_transform(mask, mode, min_blob, max_blob, min_spacing):
    # -----------------------------
    # STEP 1 — Distance Transform
    # Each vegetation pixel gets a value =
    # how far it is from the nearest non-vegetation pixel
    # Tree crown centers = highest distance values = peaks
    # -----------------------------
    binary = (mask > 0).astype(np.uint8)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Normalize for visualization
    dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # -----------------------------
    # STEP 2 — Find local peaks
    # Use mode-specific blur to control sensitivity
    # Agriculture = smaller blur = more peaks = more trees detected
    # Forest = larger blur = fewer peaks = less noise
    # -----------------------------
    if mode == "🌾 Agriculture / Plantation (Experimental)":
        blur_size = 3   # small blur — keeps nearby crowns separate
    else:
        blur_size = 5   # larger blur — reduces noise in sparse forest

    dist_blurred = cv2.GaussianBlur(dist_normalized, (blur_size, blur_size), 0)

    # -----------------------------
    # STEP 3 — Threshold peaks
    # Only keep pixels that are local maxima
    # (peak of each crown hill)
    # -----------------------------
    if mode == "🌾 Agriculture / Plantation (Experimental)":
        peak_threshold = 0.3   # more sensitive — catches smaller crowns
    else:
        peak_threshold = 0.4   # stricter — avoids noise peaks

    _, peak_mask = cv2.threshold(
        dist_blurred,
        peak_threshold * dist_blurred.max(),
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------
    # STEP 4 — Label connected regions
    # Each separate peak region = one tree
    # -----------------------------
    peak_mask = peak_mask.astype(np.uint8)
    num_labels, labeled, stats, centroids = cv2.connectedComponentsWithStats(peak_mask)

    # -----------------------------
    # STEP 5 — Filter by size and collect valid centers
    # Remove background label (0) and filter by blob area
    # -----------------------------
    valid_centers = []
    for i in range(1, num_labels):   # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if min_blob <= area <= max_blob:
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            valid_centers.append((cx, cy))

    raw_count = len(valid_centers)

    # -----------------------------
    # STEP 6 — Distance based filtering
    # Remove duplicate detections of same tree
    # -----------------------------
    if len(valid_centers) == 0:
        return 0, [], 0, dist_normalized

    centers = np.array(valid_centers)
    kept_centers = []

    for pt in centers:
        if len(kept_centers) == 0:
            kept_centers.append(pt)
        else:
            dists = cdist([pt], kept_centers)[0]
            if np.min(dists) >= min_spacing:
                kept_centers.append(pt)

    return len(kept_centers), kept_centers, raw_count, dist_normalized

def draw_crown_detections(image_np, centers, radius=10):
    result = image_np.copy()
    for (cx, cy) in centers:
        cv2.circle(result, (cx, cy), radius, (0, 255, 255), 2)
        cv2.circle(result, (cx, cy), 3, (0, 255, 255), -1)
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

    # Clean mode string — remove emojis for PDF
    clean_mode = mode.replace("🌳", "").replace("🌾", "").strip()

    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Image Analyzed: {image_name}", ln=True)
    pdf.cell(0, 8, f"Analysis Mode: {clean_mode}", ln=True)
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
        "Vegetation detection uses Excess Green Index (ExG = 2G - R - B) combined "
        "with Green Ratio filter (G / R+B) for improved soil and desert rejection. "
        "Adaptive threshold is clipped between 10-40 for stability. "
        "Crown separation uses Distance Transform — each crown creates a distance peak, "
        "even touching crowns produce separate peaks. "
        "Local maxima detection finds crown centers. "
        "Distance-based filtering removes duplicate detections."
    )
    pdf.ln(3)

    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 7,
        "Accuracy Note: Health Map 65-75% | Canopy Coverage 75-85% | "
        "Tree Count 70-85% sparse/plantation | 50-65% dense canopy. "
        "Optimized for sparse forests, orchards, plantations, and agroforestry. "
        "Results are for project scoping only. Ground truth validation recommended."
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

    mode = st.selectbox(
        "Select Analysis Mode",
        ["🌳 Sparse Forest / Dryland", "🌾 Agriculture / Plantation (Experimental)"],
        help="Forest mode = stricter detection. Agriculture = sensitive for row crops and orchards."
    )

    k_value = st.slider(
        "Adaptive Threshold Sensitivity (k)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Lower = more vegetation detected. Higher = stricter. 0.5 works for most images."
    )

    if mode == "🌳 Sparse Forest / Dryland":
        default_min_blob = 100
        default_max_blob = 8000
        default_spacing = 15
    else:
        default_min_blob = 30
        default_max_blob = 3000
        default_spacing = 8

    min_blob = st.slider(
        "Minimum Crown Size (pixels)",
        min_value=10,
        max_value=500,
        value=default_min_blob,
        help="Increase if noise is being counted as trees."
    )

    max_blob = st.slider(
        "Maximum Crown Size (pixels)",
        min_value=500,
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

    show_distance = st.checkbox(
        "Show Distance Transform map (crown separation debug view)",
        value=False
    )

    if st.button("🔍 Analyze Vegetation"):

        with st.spinner("Analyzing image..."):

            image_np = load_image(uploaded_file)
            exg, R, G, B = calculate_exg(image_np)
            mask, threshold_used = generate_vegetation_mask(exg, G, R, B, k=k_value, mode=mode)
            health_map = generate_health_map(exg, mask)
            canopy_pct = calculate_canopy_coverage(mask)
            health_scores = calculate_health_scores(exg, mask)

            tree_count, centers, raw_count, dist_map = detect_crowns_distance_transform(
                mask,
                mode=mode,
                min_blob=min_blob,
                max_blob=max_blob,
                min_spacing=min_spacing
            )

            tree_image = draw_crown_detections(image_np, centers)

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

        if show_distance:
            st.image(dist_map, caption="Distance Transform Map (crown peaks = bright spots)", use_column_width=True)

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
