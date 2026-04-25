# 🐶🐱 Cats vs Dogs Image Classifier (Deep Learning App)

A deep learning web application built with TensorFlow and Streamlit that classifies images as either a **Cat 🐱 or Dog 🐶** using a pretrained MobileNetV2 model.

---

## 🚀 Live Demo
Upload an image and get instant prediction with confidence score.

---

## 🧠 Model Architecture
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Transfer Learning applied
- Global Average Pooling layer
- Dense classification layer (Softmax)

---

## 🛠️ Tech Stack
- Python 🐍
- TensorFlow / Keras 🤖
- Streamlit 🎈
- NumPy
- Pillow

---

## 📸 Features
- Upload image from device
- Real-time prediction
- Confidence score
- Clean dark UI
- Glassmorphism design
- Lightweight and fast inference

---

## 📦 Model Info
- Input size: 224x224
- Preprocessing: Rescaling (0–1)
- Output: Cat 🐱 or Dog 🐶

---

## 📜 License
This project is licensed under GPL-3.0 — see LICENSE file for details.

---

## ⚙️ How to Run Locally

```bash
git clone <your-repo>
cd project-folder
pip install -r requirements.txt
streamlit run streamlit.py
