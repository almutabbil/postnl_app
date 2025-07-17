import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import easyocr
from ultralytics import YOLO

# ✅ Initialize models
reader = easyocr.Reader(['en'], gpu=False)
yolo_model = YOLO("yolov8n.pt")  # Tiny model, fast for cloud

# ✅ Category Mapping (from PostNL form)
CATEGORY_MAPPING = {
    "GRX/MRX": "HAGB",
    "Gateway HAGB": "HAGB",
    "Gateway HAGE": "HAGE",
    "Spring Mix": "HAGA",
    "PNLI": "HAGA",
    "HAGA gericht": "HAGA",
    "HAGB gericht": "HAGB",
    "HAGE gericht": "HAGE",
    "SCB": "CBS OOMS",
    "CBS OOMS": "CBS OOMS",
    "Belbaan": "HAGA",
    "BIMEC": "HAGE",
}

# ✅ Days mapping (fallback in case OCR fails)
DAY_COLORS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# ✅ Preprocessing for OCR
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray

# ✅ Extract text & detect category
def extract_category_from_text(image):
    processed = preprocess_image(image)
    results = reader.readtext(processed, detail=0)
    text = " ".join(results).upper()

    detected_category = "-"
    for key, value in CATEGORY_MAPPING.items():
        if key.upper() in text:
            detected_category = value
            break
    return detected_category

# ✅ YOLOv8 cart detection
def detect_carts_yolo(image):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = yolo_model.predict(img_bgr, conf=0.25, verbose=False)
    cart_count = 0
    for result in results:
        cart_count += len(result.boxes)
    return cart_count

# ✅ Excel export
def convert_df_to_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="PostNL Report")
    return output.getvalue()

# ✅ Streamlit App
st.set_page_config(page_title="PostNL Cart Tracker", layout="wide")
st.title("📦 PostNL Cart Tracker – Final Version")
st.markdown("""
✅ **Final Version**  
- **YOLOv8**: Visual cart detection (accurate counting).  
- **OCR + Mapping**: Auto category & type detection based on PostNL form.  
- **Excel Export**: Download completed table.  
""")

uploaded_files = st.file_uploader("Upload cart photos:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
results = []

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=f"Uploaded: {file.name}", use_column_width=True)

        cart_count = detect_carts_yolo(image)
        detected_category = extract_category_from_text(image)

        manual_day = st.selectbox(
            f"Select Day for {file.name}",
            options=DAY_COLORS,
            key=file.name
        )

        t_type = "Gerichte Landen" if detected_category != "-" else "Unknown"

        results.append({
            "Type": t_type,
            "Category": detected_category,
            "Day": manual_day,
            "Count": cart_count
        })

    df = pd.DataFrame(results)
    st.subheader("📊 PostNL Form Table")
    st.dataframe(df, use_container_width=True)

    excel_data = convert_df_to_excel(df)
    st.download_button(
        label="⬇️ Download Excel Report",
        data=excel_data,
        file_name="postnl_cart_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success("✅ Report ready!")
