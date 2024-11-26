import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the pre-trained model
model_path = "fashion_mnist_model.h5"  
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Fashion MNIST Classifier")
st.write("Upload a 28x28 grayscale image to classify.")

uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg"])
if uploaded_file:
    try:
        # Preprocess image
        img = Image.open(uploaded_file).convert("L").resize((28, 28))
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
        
        # Prediction
        predictions = model.predict(img_array)
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
        ]
        predicted_class = class_names[np.argmax(predictions)]
        
        # Display result
        st.image(img, caption=f"Uploaded Image", use_column_width=True)
        st.write(f"### Predicted Class: *{predicted_class}*")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
