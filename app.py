import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import easyocr
from ultralytics import YOLO
import os

# ✅ Initialize OCR and YOLO once
reader = easyocr.Reader(['en'], gpu=False)
model = YOLO("yolov8n.pt")  # Small YOLOv8 pre-trained model

# ✅ Day color mapping (fine-tune if needed)
DAY_COLORS = {
    "Sunday": (150, 75, 0),
    "Monday": (0, 128, 0),
    "Tuesday": (128, 128, 128),
    "Wednesday": (255, 255, 0),
    "Thursday": (255, 0, 0),
    "Friday": (0, 0, 255),
    "Saturday": (255, 165, 0)
}

# ✅ OCR Categories Reference
CATEGORIES = ["HAGA", "HAGB", "HAGE", "SMO", "BIMIC", "PNLI", "GRX", "MRX", "Gateway"]

# ✅ Ensure Excel exists
EXCEL_FILE = "postnl_report.xlsx"
if not os.path.exists(EXCEL_FILE):
    pd.DataFrame(columns=["Type", "Category", "Day", "Count"]).to_excel(EXCEL_FILE, index=False)

# ✅ Preprocess image for OCR
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray

# ✅ Detect day based on color cards
def detect_day_from_color(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    avg_color = img.mean(axis=(0, 1))[::-1]  # RGB
    closest_day, min_dist = "Unknown", float("inf")
    for day, color in DAY_COLORS.items():
        dist = np.linalg.norm(np.array(avg_color) - np.array(color))
        if dist < min_dist:
            closest_day, min_dist = day, dist
    return closest_day

# ✅ Extract text for category detection
def extract_category_text(image):
    processed = preprocess_image(image)
    results = reader.readtext(processed, detail=0)
    text_upper = " ".join(results).upper()
    found_category = next((c for c in CATEGORIES if c in text_upper), "Unknown")
    return found_category

# ✅ Detect carts visually using YOLOv8
def detect_carts(image):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model.predict(img_bgr, verbose=False)
    count = len(results[0].boxes) if results and len(results[0].boxes) else 0
    return count

# ✅ Append new data to Excel
def update_excel(new_data):
    df = pd.read_excel(EXCEL_FILE)
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)
    return df

# ✅ Streamlit App UI
st.set_page_config(page_title="PostNL Cart Tracker", layout="wide")
st.title("📦 PostNL Cart Tracker (Full Automated Version)")

st.markdown("""
✅ **Fully Automated Version**  
- Detects **carts visually with AI (YOLOv8)**  
- Extracts **Category via OCR**  
- Detects **Day automatically by color**  
- Saves to Excel **incrementally after each photo**  
""")

uploaded_files = st.file_uploader("Upload cart photos:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)

        # Run detections
        detected_day = detect_day_from_color(image)
        detected_category = extract_category_text(image)
        cart_count = detect_carts(image)

        # Determine Type based on category
        t_type = "Gerichte Landen" if detected_category in ["HAGA", "HAGB", "HAGE", "SMO", "BIMIC"] else "Mixed Post"

        # Save & update Excel
        new_entry = {"Type": t_type, "Category": detected_category, "Day": detected_day, "Count": cart_count}
        df = update_excel(new_entry)

        # Display results
        st.subheader(f"Results for {file.name}")
        st.image(image, caption=f"Detected Day: {detected_day} | Count: {cart_count}", use_column_width=True)
        st.dataframe(df.tail(10), use_container_width=True)

    # Download updated Excel
    with open(EXCEL_FILE, "rb") as f:
        st.download_button("⬇️ Download Updated Excel Report", f, file_name=EXCEL_FILE, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.success("✅ Report updated and saved successfully!")
