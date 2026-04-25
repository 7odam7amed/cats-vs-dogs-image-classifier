import app as st
import numpy as np
import tensorflow as tf
from PIL import Image


st.markdown("""
<style>
.glass {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)


# تحميل الموديل
model = tf.keras.models.load_model("cats_dogs_model.keras")

st.markdown("<h1 style='text-align: center;'>🐶🐱 AI Classifier</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    image = Image.open(uploaded_file)
    

    st.image(image, width="stretch")

    # preprocessing
    img = image.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    

    with st.spinner("Thinking... 🤖"):
        prediction = model.predict(img_array)
        label = np.argmax(prediction)


    result_text = "Dog 🐶" if label == 1 else "Cat 🐱"

    st.markdown(f"""
    <div class="glass">
        <h2>🧠 Prediction</h2>
        <h3>{result_text}</h3>
    </div>
    """, unsafe_allow_html=True)