import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import easyocr

# ✅ Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

st.set_page_config(page_title="PostNL Cart Tracker", layout="wide")
st.title("📦 PostNL Cart Tracker (Enhanced Version)")

st.markdown("""
✅ **Enhanced Version**  
- Detects **number of carts visually** (OpenCV).  
- Improved **table output (based on provided form)**.  
- Works on **Streamlit Cloud**.  
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

# ✅ Cart Detection (OpenCV)
def detect_carts(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cart_count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 0.7 < aspect_ratio < 1.5 and w > 100 and h > 100:
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
        cart_count = detect_carts(image)

        results.append({
            "Type": t_type,
            "Category": t_category,
            "Day": "Unknown",  # Can still use your color-detection logic if needed
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
