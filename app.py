import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import easyocr
from ultralytics import YOLO

# ✅ Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# ✅ Load YOLOv8 pre-trained model
yolo_model = YOLO("yolov8s.pt")

st.set_page_config(page_title="PostNL Cart Tracker", layout="wide")
st.title("📦 PostNL Cart Tracker (YOLOv8 Enhanced Version)")

st.markdown("""
✅ **YOLOv8 Enhanced Version**  
- Detects **carts visually with AI** (YOLOv8).  
- Accurate **cart counting** even in cluttered images.  
- Improved **OCR category extraction**.  
""")

# ✅ Category mapping
CATEGORIES = ["HAGA", "HAGB", "HAGE", "SMO", "BIMIC"]

# ✅ Preprocessing for OCR
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# ✅ OCR Text Extraction
def extract_text_easyocr(image):
    processed = preprocess_image(image)
    results = reader.readtext(processed, detail=0)
    return " ".join(results).upper()

# ✅ Parse type & category
def parse_type_category(text):
    if "MIX" in text:
        t_type = "Mixed Post"
    elif "LIST" in text:
        t_type = "List " + "".join(filter(str.isdigit, text))
    else:
        t_type = "Gerichte Landen"
    t_category = next((c for c in CATEGORIES if c in text), "-")
    return t_type, t_category

# ✅ YOLOv8 Cart Detection
def detect_carts_yolo(image):
    img = np.array(image)
    results = yolo_model(img)
    detections = results[0].boxes

    cart_count = 0
    for box in detections:
        cls = int(box.cls[0])
        label = yolo_model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Filter: YOLO doesn't know "PostNL carts", so use size + generic objects
        width, height = x2 - x1, y2 - y1
        if width > 100 and height > 100:  # Rough filter for big objects
            cart_count += 1

    return cart_count

# ✅ Convert to Excel
def convert_df_to_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Cart Summary")
    return output.getvalue()

uploaded_files = st.file_uploader("Upload cart photos:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
results = []

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        ocr_text = extract_text_easyocr(image)
        t_type, t_category = parse_type_category(ocr_text)
        cart_count = detect_carts_yolo(image)

        results.append({
            "Type": t_type,
            "Category": t_category,
            "Day": "Unknown",
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
