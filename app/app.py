import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import json
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL & LABELS
# =========================
model = tf.keras.models.load_model("road_damage_cnn.keras")

with open("label_map.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = 128

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Road Damage AI", layout="wide")

# =========================
# SECTION 1 — HEADER
# =========================
st.markdown("""
# 🚧 AI-Based Road Damage Detection System  
### Smart City Infrastructure Monitoring using CNN
""")

st.divider()

# =========================
# SECTION 2 — ABOUT PROJECT
# =========================
with st.expander("📘 About the Project", expanded=False):
    st.write("""
    Road monitoring is essential for ensuring safe transportation and reducing accidents caused by potholes,
    cracks, and structural road damage.

    CNN (Convolutional Neural Networks) is widely used in computer vision to automatically extract features
    from images and classify them accurately without manual feature engineering.

    ### Industry Applications:
    - Smart city infrastructure monitoring  
    - Autonomous vehicle safety systems  
    - Highway maintenance automation  
    - Government road inspection systems  
    """)

st.divider()

# =========================
# SECTION 3 — UPLOAD AREA
# =========================
st.subheader("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload an image of road damage",
    type=["jpg", "png", "jpeg",'avif']
)

# =========================
# PROCESS IMAGE
# =========================
if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file)

    # =========================
    # SECTION 4 — IMAGE PREVIEW
    # =========================
    st.subheader("🖼️ Uploaded Image Preview")
    st.image(image, use_container_width=True)

    # Preprocess
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img_input = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img_input)[0]
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    label = class_names[class_index]

    # Severity logic
    if label == "pothole":
        severity = "🔴 High"
        recommendation = "Immediate maintenance required."
    elif label == "crack":
        severity = "🟠 Medium"
        recommendation = "Schedule repair soon."
    else:
        severity = "🟡 Low"
        recommendation = "Monitor condition periodically."

    st.divider()

    # =========================
    # SECTION 5 — PREDICTION AREA
    # =========================
    st.subheader("🧠 Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction", label)
    col2.metric("Confidence", f"{confidence*100:.2f}%")
    col3.metric("Severity", severity)

    st.success(recommendation)

    st.divider()

    # =========================
    # SECTION 6 — VISUALIZATION
    # =========================
    st.subheader("📊 Class Probability Visualization")

    fig, ax = plt.subplots()
    ax.bar(class_names, prediction, color=["#ff4b4b", "#ffa500", "#4caf50"])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)

    st.divider()

    # =========================
    # SECTION 7 — RECOMMENDATIONS
    # =========================
    st.subheader("⚠️ Safety Recommendations")

    if label == "pothole":
        st.error("""
        🚨 Immediate maintenance recommended  
        ⚠️ High-risk road condition  
        🛑 Can cause vehicle damage and accidents
        """)
    elif label == "crack":
        st.warning("""
        ⚠️ Preventive maintenance suggested  
        🛠️ Monitor for expansion  
        """)
    else:
        st.info("""
        ✅ Low risk detected  
        🟢 No immediate action required  
        """)

else:
    st.info("👆 Upload a road image to start prediction.")