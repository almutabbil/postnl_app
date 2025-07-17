import streamlit as st
import pandas as pd
import cv2
import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# =============================
# 1. SETUP
# =============================

# Load YOLOv8 model for cart detection (trained on carts or general objects)
MODEL_PATH = "yolov8n.pt"  # Using the small model to ensure Streamlit Cloud compatibility
model = YOLO(MODEL_PATH)

# EasyOCR reader for text/category extraction
reader = easyocr.Reader(['en'])

# Excel file to save data
EXCEL_FILE = "postnl_report.xlsx"

# Initialize Excel if not exists
if not os.path.exists(EXCEL_FILE):
    df = pd.DataFrame(columns=["Type", "Category", "Day", "Count"])
    df.to_excel(EXCEL_FILE, index=False)

# =============================
# 2. FUNCTIONS
# =============================

def detect_carts(image):
    """Detect carts using YOLOv8 and return total count."""
    results = model.predict(image, conf=0.3, verbose=False)
    count = 0
    for r in results:
        for box in r.boxes:
            count += 1
    return count

def extract_text_category(image):
    """Extract category text using EasyOCR."""
    img_array = np.array(image)
    results = reader.readtext(img_array)
    text_results = [res[1] for res in results]
    
    category = "-"
    type_detected = "Unknown"
    
    # Match against known words
    known_types = ["Gerichte", "HAGA", "HAGB", "HAGE", "Gateway", "Spring", "PNLI", "Belbaan"]
    for t in known_types:
        if any(t.lower() in tx.lower() for tx in text_results):
            type_detected = t
            category = t
            break
    
    return type_detected, category

def save_to_excel(type_detected, category, day, count):
    """Append detection result to the Excel file."""
    df = pd.read_excel(EXCEL_FILE)
    new_row = {"Type": type_detected, "Category": category, "Day": day, "Count": count}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)

# =============================
# 3. STREAMLIT INTERFACE
# =============================

st.set_page_config(page_title="PostNL Cart Scanner", layout="wide")
st.title("📦 PostNL Cart Scanner - Automated Counting")

uploaded_file = st.file_uploader("Upload a photo of carts", type=["jpg", "jpeg", "png"])

day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    selected_day = st.selectbox("Select day (auto selection coming soon)", day_options)

    if st.button("🔍 Analyze Photo"):
        with st.spinner("Analyzing... please wait"):
            cart_count = detect_carts(image)
            type_detected, category = extract_text_category(image)
            save_to_excel(type_detected, category, selected_day, cart_count)
        
        st.success(f"✅ Detected: {cart_count} carts | Type: {type_detected} | Day: {selected_day}")
        st.dataframe(pd.read_excel(EXCEL_FILE))

        with open(EXCEL_FILE, "rb") as f:
            st.download_button("⬇ Download Updated Excel", f, file_name=EXCEL_FILE)
