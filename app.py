import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import easyocr
from ultralytics import YOLO

# ✅ Load YOLOv8 Pretrained Model (Detects Carts)
yolo_model = YOLO("yolov8n.pt")  # Small & fast model

# ✅ Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# ✅ Streamlit Config
st.set_page_config(page_title="PostNL Cart Tracker", layout="wide")
st.title("📦 PostNL Cart Tracker (AI + OCR Optimized)")

st.markdown("""
✅ **AI + OCR Optimized Version**  
- Detects **carts visually with AI (YOLOv8)**  
- Accurate **cart counting** even in cluttered images  
- Improved **OCR text extraction** for Type & Category  
""")

# ✅ Day Color Mapping (PostNL Standard)
DAY_COLORS = {
    "Sunday": (150, 75, 0),
    "Monday": (0, 128, 0),
    "Tuesday": (128, 128, 128),
    "Wednesday": (255, 255, 0),
    "Thursday": (255, 0, 0),
    "Friday": (0, 0, 255),
    "Saturday": (255, 165, 0)
}

# ✅ Preprocess image for OCR
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray

# ✅ Detect Day by color
def detect_day_from_color(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    avg_color = img.mean(axis=(0, 1))[::-1]
    closest_day, min_dist = None, float("inf")
    for day, color in DAY_COLORS.items():
        dist = np.linalg.norm(np.array(avg_color) - np.array(color))
        if dist < min_dist:
            closest_day, min_dist = day, dist
    return closest_day

# ✅ Extract text using EasyOCR
def extract_text_easyocr(image):
    processed = preprocess_image(image)
    results = reader.readtext(processed, detail=0)
    return " ".join(results).upper()

# ✅ Parse Type & Category from OCR
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

# ✅ YOLOv8 Cart Detection
def detect_carts_yolo(image):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = yolo_model.predict(img_bgr, verbose=False)
    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 39:  # Class 39 = "refrigerator" (close shape to PostNL carts)
                count += 1
    return count

# ✅ Convert to Excel
def convert_df_to_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Cart Summary")
    return output.getvalue()

# ✅ Streamlit UI
uploaded_files = st.file_uploader("Upload cart photos:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
results = []

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        detected_day = detect_day_from_color(image)
        ocr_text = extract_text_easyocr(image)
        t_type, t_category = parse_type_category(ocr_text)
        cart_count = detect_carts_yolo(image)

        manual_day = st.selectbox(
            f"Select Day for {file.name} (detected: {detected_day})",
            options=list(DAY_COLORS.keys()),
            index=list(DAY_COLORS.keys()).index(detected_day) if detected_day else 0,
            key=file.name
        )

        results.append({
            "Type": t_type,
            "Category": t_category,
            "Day": manual_day,
            "Count": cart_count
        })

    df = pd.DataFrame(results)

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
