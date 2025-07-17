import os
import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import easyocr

# ✅ Fix for EasyOCR on Streamlit Cloud (forces model to download in /tmp)
os.environ["TORCH_HOME"] = "/tmp/torch"

# ✅ Initialize EasyOCR reader (English only for faster processing)
reader = easyocr.Reader(['en'], gpu=False)

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="PostNL Cart Tracker", layout="wide")
st.title("📦 PostNL Cart Tracker (Cloud-Optimized Version)")

st.markdown("""
✅ **Cloud-Optimized Version**  
- Works on **Streamlit Cloud & iPhone**  
- Detects **multiple carts per photo**  
- Improved **day color detection**  
- **EasyOCR optimized** (no Tesseract needed)
""")

# ✅ Day Color Mapping
DAY_COLORS = {
    "Sunday": (150, 75, 0),
    "Monday": (0, 128, 0),
    "Tuesday": (128, 128, 128),
    "Wednesday": (255, 255, 0),
    "Thursday": (255, 0, 0),
    "Friday": (0, 0, 255),
    "Saturday": (255, 165, 0)
}

# ✅ Preprocessing (Cloud Safe & Memory Optimized)
def preprocess_image(image):
    """Resize, grayscale, and sharpen image for better OCR (Cloud Safe)."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Resize if too large (Streamlit Cloud memory-safe)
    max_dim = 1200
    if img.shape[0] > max_dim or img.shape[1] > max_dim:
        scaling_factor = max_dim / max(img.shape[0], img.shape[1])
        img = cv2.resize(img, (int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray

# ✅ Detect Day Based on Average Color
def detect_day_from_color(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    avg_color = img.mean(axis=(0, 1))[::-1]  # Convert BGR → RGB

    closest_day, min_dist = None, float("inf")
    for day, color in DAY_COLORS.items():
        dist = np.linalg.norm(np.array(avg_color) - np.array(color))
        if dist < min_dist:
            closest_day, min_dist = day, dist
    return closest_day

# ✅ OCR Extraction with EasyOCR
def extract_text_easyocr(image):
    processed = preprocess_image(image)
    results = reader.readtext(processed, detail=0)
    return " ".join(results).upper()

# ✅ Type & Category Parsing
def parse_type_category(text):
    if "MIX" in text:
        t_type = "Mixed Post"
    elif "LIST" in text:
        t_type = "List " + "".join(filter(str.isdigit, text))
    else:
        t_type = "Gerichte Landen"

    categories = ["HAGA", "HAGB", "HAGE", "SMO", "BIMIC"]
    t_category = next((c for c in categories if c in text), "-")
    return t_type, t_category

# ✅ Convert DataFrame to Excel
def convert_df_to_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Cart Summary")
    return output.getvalue()

# ✅ File Upload & Processing
uploaded_files = st.file_uploader("Upload cart photos:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
results = []

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        ocr_text = extract_text_easyocr(image)
        detected_day = detect_day_from_color(image)
        t_type, t_category = parse_type_category(ocr_text)

        manual_day = st.selectbox(
            f"Select Day for {file.name} (detected: {detected_day})",
            options=list(DAY_COLORS.keys()),
            index=list(DAY_COLORS.keys()).index(detected_day) if detected_day else 0,
            key=file.name
        )

        results.append({"Type": t_type, "Category": t_category, "Day": manual_day, "Count": 1})

    df = pd.DataFrame(results).groupby(["Type", "Category", "Day"], as_index=False)["Count"].sum()

    st.subheader("📊 Cart Summary Table")
    st.dataframe(df, use_container_width=True)

    excel_data = convert_df_to_excel(df)
    st.download_button(
        label="⬇️ Download Excel Report",
        data=excel_data,
        file_name="postnl_cart_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success("✅ Report ready!")
