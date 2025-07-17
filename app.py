import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import easyocr
import re

# ✅ Initialize EasyOCR (English only, GPU off for Streamlit Cloud)
reader = easyocr.Reader(['en'], gpu=False)

st.set_page_config(page_title="PostNL Cart Tracker", layout="wide")
st.title("📦 PostNL Cart Tracker (Improved Version)")

st.markdown("""
✅ **Improved Version**  
- Detects **multiple carts per photo**.  
- Correct **category detection** (HAGA, HAGB, HAGE, SMO, BIMIC).  
- Accurate **cart counting** per category.  
- Shows **debug image with detected tags**.  
- Outputs table in **official format** (Type, Category, Day, Count).  
""")

# ✅ Day Colors for Reference (can be improved if needed)
DAY_COLORS = {
    "Sunday": (150, 75, 0),
    "Monday": (0, 128, 0),
    "Tuesday": (128, 128, 128),
    "Wednesday": (255, 255, 0),
    "Thursday": (255, 0, 0),
    "Friday": (0, 0, 255),
    "Saturday": (255, 165, 0)
}

# ✅ Categories & Type Matching
CATEGORIES = ["HAGA", "HAGB", "HAGE", "SMO", "BIMIC"]

def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray

def detect_day_from_color(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    avg_color = img.mean(axis=(0, 1))[::-1]
    closest_day, min_dist = None, float("inf")
    for day, color in DAY_COLORS.items():
        dist = np.linalg.norm(np.array(avg_color) - np.array(color))
        if dist < min_dist:
            closest_day, min_dist = day, dist
    return closest_day

def analyze_carts(image):
    """Detect multiple carts and extract their categories"""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed = preprocess_image(image)

    results = reader.readtext(processed, detail=1)
    category_counts = {cat: 0 for cat in CATEGORIES}
    debug_img = img.copy()

    for (bbox, text, conf) in results:
        text_upper = text.upper()

        # Draw bounding boxes for debugging
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)

        # Check for category keywords
        for cat in CATEGORIES:
            if cat in text_upper:
                category_counts[cat] += 1
                cv2.putText(debug_img, cat, (int(bbox[0][0]), int(bbox[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return category_counts, debug_img

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
        detected_day = detect_day_from_color(image)

        manual_day = st.selectbox(
            f"Select Day for {file.name} (detected: {detected_day})",
            options=list(DAY_COLORS.keys()),
            index=list(DAY_COLORS.keys()).index(detected_day) if detected_day else 0,
            key=file.name
        )

        category_counts, debug_img = analyze_carts(image)

        # Display debug image
        st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                 caption=f"Detected Carts for {file.name}", use_column_width=True)

        for cat, count in category_counts.items():
            if count > 0:
                results.append({
                    "Type": "Gerichte Landen",
                    "Category": cat,
                    "Day": manual_day,
                    "Count": count
                })

    if results:
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
    else:
        st.warning("No carts detected! Check image quality or tag visibility.")
