import streamlit as st
import pandas as pd
import easyocr
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Initialize EasyOCR (Dutch + English)
reader = easyocr.Reader(['nl', 'en'], gpu=False)

st.set_page_config(page_title="PostNL Cart Tracker (Form Version)", layout="wide")
st.title("📦 PostNL Cart Tracker – Form Template Version")

st.write("""
✅ **Optimized for your official PostNL Form**  
- Better OCR accuracy with preprocessing.  
- Auto-fills the same columns as the printed form.  
- Exports to Excel in the official layout.  
""")

# Day color reference
DAY_COLORS = {
    "Sunday": (150, 75, 0),      # Brown
    "Monday": (0, 128, 0),       # Green
    "Tuesday": (128, 128, 128),  # Grey
    "Wednesday": (255, 255, 0),  # Yellow
    "Thursday": (255, 0, 0),     # Red
    "Friday": (0, 0, 255),       # Blue
    "Saturday": (255, 165, 0)    # Orange
}

def preprocess_image(image):
    """Convert to grayscale + threshold to improve OCR."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh

def detect_day_from_color(image):
    """Detect day based on card color."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    avg_color = img.mean(axis=(0, 1))[::-1]  # RGB
    closest_day, min_dist = None, float("inf")
    for day, color in DAY_COLORS.items():
        dist = np.linalg.norm(np.array(avg_color) - np.array(color))
        if dist < min_dist:
            min_dist, closest_day = dist, day
    return closest_day

def extract_text(image):
    """OCR with EasyOCR."""
    result = reader.readtext(preprocess_image(image), detail=0, paragraph=True)
    return " ".join(result).upper()

def parse_type_category(text):
    """Map text to Type and Category."""
    if "MIX" in text:
        t_type = "Mixed Post"
    elif "LIST" in text:
        t_type = "List " + "".join(filter(str.isdigit, text))
    else:
        t_type = "Gerichte Landen"

    categories = ["HAGA", "HAGB", "HAGE", "SMO", "BIMIC", "GRZAMRX", "RC"]
    t_category = next((cat for cat in categories if cat in text), "-")
    return t_type, t_category

uploaded_files = st.file_uploader("Upload cart photos:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

results = []

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        ocr_text = extract_text(image)
        detected_day = detect_day_from_color(image)
        t_type, t_category = parse_type_category(ocr_text)

        manual_day = st.selectbox(
            f"Select Day for {file.name} (detected: {detected_day})",
            options=list(DAY_COLORS.keys()),
            index=list(DAY_COLORS.keys()).index(detected_day) if detected_day else 0,
            key=file.name
        )

        results.append({
            "Voorraad startproduct": t_type,
            "RC": t_category,
            "P1ats": manual_day,
            "Opmerkingen": ""
        })

    df = pd.DataFrame(results)

    st.subheader("📊 PostNL Form Table")
    st.dataframe(df, use_container_width=True)

    def convert_df_to_excel(dataframe):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            dataframe.to_excel(writer, index=False, sheet_name="PostNL Form")
        return output.getvalue()

    excel_data = convert_df_to_excel(df)
    st.download_button(
        label="⬇️ Download PostNL Form Excel",
        data=excel_data,
        file_name="postnl_form.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success("✅ Form Ready!")
