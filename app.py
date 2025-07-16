import streamlit as st
import pandas as pd
import easyocr
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

reader = easyocr.Reader(['en', 'nl'], gpu=False)

st.set_page_config(page_title="PostNL Cart Tracker", layout="wide")
st.title("📦 PostNL Cart Tracker (Cloud-Ready Version)")

st.write("""
✅ **Cloud-Optimized Version**
- Uses **EasyOCR** (works without Tesseract).
- Detects **multiple carts per photo**.
- Improved day color detection.
- Works on **Streamlit Cloud** & iPhone.
""")

DAY_COLORS = {
    "Sunday": (150, 75, 0),
    "Monday": (0, 128, 0),
    "Tuesday": (128, 128, 128),
    "Wednesday": (255, 255, 0),
    "Thursday": (255, 0, 0),
    "Friday": (0, 0, 255),
    "Saturday": (255, 165, 0)
}

def detect_day_from_color(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    avg_color = img.mean(axis=(0, 1))[::-1]
    closest_day, min_dist = None, float("inf")
    for day, color in DAY_COLORS.items():
        dist = np.linalg.norm(np.array(avg_color) - np.array(color))
        if dist < min_dist:
            min_dist, closest_day = dist, day
    return closest_day

def resize_image(image, max_size=1200):
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        return image.resize((int(w*scale), int(h*scale)))
    return image

def extract_text_easyocr(image):
    img = np.array(image)
    results = reader.readtext(img, detail=0)
    return [r.strip().upper() for r in results if len(r.strip()) > 2]

def parse_type_category(text):
    if "MIX" in text:
        t_type = "Mixed Post"
    elif "LIST" in text:
        numbers = "".join(filter(str.isdigit, text))
        t_type = f"List {numbers}" if numbers else "List"
    else:
        t_type = "Gerichte Landen"
    possible_categories = ["HAGA", "HAGB", "HAGE", "SMO", "BIMIC"]
    t_category = "-"
    for cat in possible_categories:
        if cat in text:
            t_category = cat
            break
    return t_type, t_category

uploaded_files = st.file_uploader("Upload cart photos:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
results = []

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        image = resize_image(image)
        detected_day = detect_day_from_color(image)
        texts = extract_text_easyocr(image)
        for idx, text in enumerate(texts):
            t_type, t_category = parse_type_category(text)
            manual_day = st.selectbox(
                f"Select Day for {file.name} - cart {idx+1} (detected: {detected_day})",
                options=list(DAY_COLORS.keys()),
                index=list(DAY_COLORS.keys()).index(detected_day) if detected_day else 0,
                key=f"{file.name}_{idx}"
            )
            results.append({
                "Type": t_type,
                "Category": t_category,
                "Day": manual_day,
                "Count": 1
            })
    df = pd.DataFrame(results)
    df = df.groupby(["Type", "Category", "Day"], as_index=False)["Count"].sum()
    st.subheader("📊 Cart Summary Table")
    st.dataframe(df, use_container_width=True)

    def convert_df_to_excel(dataframe):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            dataframe.to_excel(writer, index=False, sheet_name="Cart Summary")
        return output.getvalue()

    excel_data = convert_df_to_excel(df)
    st.download_button(
        label="⬇️ Download Excel Report",
        data=excel_data,
        file_name="postnl_cart_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.success("✅ Report ready!")
