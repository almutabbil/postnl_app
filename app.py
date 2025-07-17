import streamlit as st

import torch
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # prevents DLL issues on cloud
import cv2
# ================================
# ✅ Initialize YOLO and EasyOCR
# ================================
MODEL_PATH = "yolov8_cart.pt"  # Replace with your trained YOLO model for carts
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

EXCEL_FILE = "postnl_report.xlsx"

# ================================
# ✅ Helper Functions
# ================================
def detect_carts(image):
    """Detect carts using YOLOv8"""
    results = model(image)
    cart_count = 0
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls].lower() == "cart":  # Ensure correct class name
                cart_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2))
    return cart_count, detections

def detect_category_and_day(image):
    """Detect text from carts and identify category & day"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ocr_results = reader.readtext(gray)

    categories = []
    day_detected = None
    possible_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    for (_, text, _) in ocr_results:
        text_clean = text.lower().strip()

        # Detect day
        for d in possible_days:
            if d in text_clean:
                day_detected = d.capitalize()

        # Detect category (basic keywords, can be expanded)
        if any(kw in text_clean for kw in ["gericht", "spring", "gateway", "reject", "pallets"]):
            categories.append(text_clean)

    # Fallback
    if not day_detected:
        day_detected = "Unknown"
    if not categories:
        categories = ["Unknown"]

    return list(set(categories)), day_detected

def update_excel(categories, day, cart_count):
    """Update the Excel report automatically"""
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
    else:
        df = pd.DataFrame(columns=["Type", "Category", "Day", "Count"])

    for category in categories:
        df = pd.concat([df, pd.DataFrame([{
            "Type": "Detected Carts",
            "Category": category,
            "Day": day,
            "Count": cart_count
        }])], ignore_index=True)

    df.to_excel(EXCEL_FILE, index=False)

# ================================
# ✅ Streamlit UI
# ================================
st.set_page_config(page_title="PostNL Cart Analyzer", layout="wide")
st.title("📦 PostNL Cart Analyzer (Auto Day & Category Detection)")

uploaded_file = st.file_uploader("Upload a cart photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Photo"):
        with st.spinner("Analyzing photo..."):
            cart_count, detections = detect_carts(image)
            categories, detected_day = detect_category_and_day(image)

            update_excel(categories, detected_day, cart_count)

            st.success("✅ Analysis complete!")
            st.write(f"**Detected Day:** {detected_day}")
            st.write(f"**Cart Count:** {cart_count}")
            st.write(f"**Categories:** {', '.join(categories)}")

            df_preview = pd.read_excel(EXCEL_FILE)
            st.dataframe(df_preview)

        with open(EXCEL_FILE, "rb") as f:
            st.download_button("⬇ Download Updated Excel", f, file_name=EXCEL_FILE)
