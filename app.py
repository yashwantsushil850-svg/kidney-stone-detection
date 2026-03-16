import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import time

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Kidney Stone Detection System",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("kidney_stone_pureCNN.keras")
        return model
    except:
        st.error("Model could not be loaded. Ensure model file exists.")
        st.stop()

model = load_model()

# ---------------------------------------------------
# Grad-CAM Function (Explainable AI)
# ---------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# ---------------------------------------------------
# Sidebar Information
# ---------------------------------------------------
st.sidebar.title("AI Model Information")

st.sidebar.markdown("""
**Model:** Convolutional Neural Network (CNN)

**Training Dataset:**  
32,264 Total Images  
17,794 Stone Images  
14,470 Non-Stone Images  

**Image Resolution:**  
128 × 128  

**Best Validation Accuracy:**  
99.05%

This AI system assists in detecting kidney stones from CT scan images.
""")

st.sidebar.info("⚠ This tool is intended for research and educational purposes.")

# ---------------------------------------------------
# Model Comparison Chart
# ---------------------------------------------------
comparison = pd.DataFrame({
    "Model":[
        "Pure CNN",
        "Batch Normalization",
        "Instance Normalization",
        "Group Normalization",
        "Layer Normalization"
    ],
    "Validation Accuracy":[
        99.05,
        98.10,
        96.68,
        97.33,
        81.12
    ]
})

st.sidebar.subheader("Model Comparison")
st.sidebar.bar_chart(comparison.set_index("Model"))

# ---------------------------------------------------
# Main Title
# ---------------------------------------------------
st.title("🩺 AI Kidney Stone Detection System")

st.markdown("""
Upload a **CT scan image** to detect the presence of kidney stones using a trained deep learning model.
""")

# ---------------------------------------------------
# Instructions
# ---------------------------------------------------
with st.expander("How to use this AI tool"):
    st.write("""
1. Upload a CT scan image of the kidney  
2. Click **Analyze Image**  
3. The AI model predicts whether a kidney stone exists  
4. AI explanation map shows where the model focused  
""")

# ---------------------------------------------------
# Upload Image
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CT Scan Image",
    type=["jpg","jpeg","png"]
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    if st.button("Analyze Image"):

        with st.spinner("Analyzing image using AI model..."):

            time.sleep(1)

            img = image.resize((128,128))
            img_array = np.array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)[0][0]

        st.subheader("Diagnosis Result")

        if prediction > 0.5:

            confidence = float(prediction)
            risk = confidence
            st.error("⚠ Kidney Stone Detected")

        else:

            confidence = float(1 - prediction)
            risk = confidence
            st.success("✔ No Kidney Stone Detected")

        st.subheader("AI Confidence Level")

        risk = max(0.0, min(risk, 1.0))
        st.progress(risk)

        st.write(f"Confidence Score: **{risk*100:.2f}%**")

        # ---------------------------------------------------
        # Grad-CAM Visualization
        # ---------------------------------------------------
        st.subheader("AI Attention Visualization (Grad-CAM)")

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            "conv2d_2"  # change if your last conv layer name is different
        )

        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * 0.4 + np.array(image)

        st.image(superimposed_img.astype("uint8"),
                 caption="Grad-CAM AI Attention Map")

        # ---------------------------------------------------
        # Download Report
        # ---------------------------------------------------
        report = f"""
Kidney Stone Detection Report
-------------------------------------

Prediction Result:
{"Kidney Stone Detected" if prediction > 0.5 else "No Kidney Stone Detected"}

Confidence Score:
{risk*100:.2f} %

Model:
Convolutional Neural Network (CNN)

Validation Accuracy:
99.05%

Explainable AI Method:
Grad-CAM Visualization

Note:
This AI analysis is intended for research and educational purposes only.
"""

        st.download_button(
            label="Download AI Diagnosis Report",
            data=report,
            file_name="kidney_stone_report.txt",
            mime="text/plain"
        )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")

st.caption("""
AI Kidney Stone Detection System  
Developed using Deep Learning (CNN) and Streamlit
""")