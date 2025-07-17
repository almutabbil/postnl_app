import streamlit as st
import pandas as pd
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image
import os

# ======================
# INITIAL SETUP
# ======================
EXCEL_FILE = "postnl_report.xlsx"
YOLO_MODEL = "yolov8n.pt"  # Use YOLOv8 nano model (fast for Streamlit)

# Load YOLOv8
model = YOLO(YOLO_MODEL)
reader = easyocr.Reader(['en'])

# Initialize Excel file if not exists
if not os.path.exists(EXCEL_FILE):
    df = pd.DataFrame(columns=["Type", "Category", "Day", "Count"])
    df.to_excel(EXCEL_FILE, index=False)

# ======================
# HELPER FUNCTIONS
# ======================
def detect_carts(image):
    results = model(image)
    cart_count = 0
    for r in results:
        for c in r.boxes.cls:
            cart_count += 1
    return cart_count

def extract_category_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray, detail=0)
    category = "Unknown"
    for text in result:
        if any(word in text.lower() for word in ["gericht", "landen", "haga", "hagb", "hagb", "spring", "belbaan"]):
            category = text
            break
    return category

def update_excel(cart_type, category, day, count):
    df = pd.read_excel(EXCEL_FILE)
    new_row = {"Type": cart_type, "Category": category, "Day": day, "Count": count}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)

# ======================
# STREAMLIT INTERFACE
# ======================
st.title("📦 PostNL Cart Detection & Reporting")
st.write("Upload a photo, select the day, and automatically count carts + update Excel.")

uploaded_file = st.file_uploader("Upload cart photo:", type=["jpg", "jpeg", "png"])

# Dropdown for day selection
day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day = st.selectbox("Select Day", day_options, index=day_options.index("Tuesday"))

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Photo"):
        with st.spinner("Processing image..."):
            count = detect_carts(img_array)
            category = extract_category_text(img_array)
            cart_type = "Gerichte Lande" if "gericht" in category.lower() else "Unknown"

            # Display results
            st.success(f"✅ Detected {count} carts | Type: {cart_type} | Category: {category} | Day: {selected_day}")

            # Update Excel
            update_excel(cart_type, category, selected_day, count)
            st.info(f"Excel updated: {EXCEL_FILE}")

        # Show updated Excel in table
        updated_df = pd.read_excel(EXCEL_FILE)
        st.dataframe(updated_df)
        st.download_button("⬇ Download Updated Excel", data=open(EXCEL_FILE, "rb"), file_name=EXCEL_FILE)
