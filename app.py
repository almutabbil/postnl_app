import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import easyocr

# ✅ Streamlit Page Setup
st.set_page_config(page_title="PostNL Cart Tracker", layout="wide")
st.title("📦 PostNL Cart Tracker (Cloud-Optimized Version)")
st.write("✅ App loaded successfully. Upload images to begin.")

# ✅ Cache EasyOCR to prevent repeated initialization
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr_reader()

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

# ✅ Functions
def preprocess_image(image):
    """Resize, grayscale, and sharpen image for better OCR."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray

def detect_day_from_color(image):
    """Improved day detection based on color averaging."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    avg_color = img.mean(axis=(0, 1))[::-1]  # RGB
    closest_day, min_dist = None, float("inf")
    for day, color in DAY_COLORS.items():
        dist = np.linalg.norm(np.array(avg_color) - np.array(color))
        if dist < min_dist:
            closest_day, min_dist = day, dist
    return closest_day

def extract_text_easyocr(image):
    """Extract text using EasyOCR with preprocessing."""
    processed = preprocess_image(image)
    results = reader.readtext(processed, detail=0)
    return " ".join(results).upper()

def parse_type_category(text):
    """Extract Type & Category from OCR text."""
    if "MIX" in text:
        t_type = "Mixed Post"
    elif "LIST" in text:
        t_type = "List " + "".join(filter(str.isdigit, text))
    else:
        t_type = "Gerichte Landen"

    categories = ["HAGA", "HAGB", "HAGE", "SMO", "BIMIC"]
    t_category = next((c for c in categories if c in text), "-")
    return t_type, t_category

def convert_df_to_excel(dataframe):
    """Export DataFrame to Excel (auto-download)."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Cart Summary")
    return output.getvalue()

# ✅ Main App Logic
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
else:
    st.info("👆 Upload one or more cart photos to start OCR processing.")
