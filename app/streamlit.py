import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.processing import load_and_preprocess_image, get_prediction_label


st.set_page_config(
    page_title="AI Classifier",
    page_icon="🐶",
    layout="wide"
)


st.markdown("""
<style>
body {
    background: linear-gradient(-45deg, #0f0f0f, #1a1a2e, #16213e, #0f0f0f);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.card {
    background: rgba(255,255,255,0.07);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}

.title {
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:white;
}
</style>
""", unsafe_allow_html=True)


model = tf.keras.models.load_model("model/cats_dogs_model.keras")


st.markdown("<div class='title'>🐶🐱 AI Vision Classifier</div>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Drop an image here", type=["jpg","png","jpeg"])


if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    image_reshape = load_and_preprocess_image(image)

    with st.spinner("AI is analyzing image... 🤖"):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.003)
            progress.progress(i+1)

        prediction = model.predict(image_reshape)

        label, confidence = get_prediction_label(prediction)

    result = "Dog 🐶" if label==1 else "Cat 🐱"
    print(result)

    with col2:
        st.markdown("### 🧠 Prediction")

        st.markdown(f"""
        <div class="card">
            <h2>{result}</h2>
            <p>Confidence: {confidence:.3f}</p>
        </div>
        """, unsafe_allow_html=True)


        st.progress(float(confidence))


    if label == 1:
        st.success("Dog detected successfully!")
    else:
        st.info("Cat detected successfully!")