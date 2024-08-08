import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input

# Load the trained models
mobilenet_model_path = 'best_model_net.keras'
densenet_model_path = 'best_model_densenet121.keras'

try:
    mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)
    st.success("MobileNetV2 model loaded successfully.")
except Exception as e:
    st.error(f"Error loading MobileNetV2 model: {e}")

try:
    densenet_model = tf.keras.models.load_model(densenet_model_path)
    st.success("DenseNet121 model loaded successfully.")
except Exception as e:
    st.error(f"Error loading DenseNet121 model: {e}")

# Function to preprocess and predict using a specified model
def preprocess_and_predict(image_path, model, preprocess_input, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    return predictions

# Function to interpret predictions
def interpret_predictions(predictions):
    real_prob, fake_prob = predictions[0]
    if real_prob > fake_prob:
        label = "Real"
        confidence = real_prob
    else:
        label = "Fake"
        confidence = fake_prob
    return label, confidence

# Streamlit app
st.title("Deepfake Detection with MobileNetV2 and DenseNet121")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Ensure the temporary directory exists
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Save the uploaded file temporarily
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Predict using MobileNetV2
    if mobilenet_model:
        st.subheader("MobileNetV2 Prediction")
        mobilenet_predictions = preprocess_and_predict(temp_file_path, mobilenet_model, mobilenet_preprocess_input, (224, 224))
        mobilenet_label, mobilenet_confidence = interpret_predictions(mobilenet_predictions)
        st.write(f"Prediction: {mobilenet_label} (Confidence: {mobilenet_confidence:.2f})")

    # Predict using DenseNet121
    if densenet_model:
        st.subheader("DenseNet121 Prediction")
        densenet_predictions = preprocess_and_predict(temp_file_path, densenet_model, densenet_preprocess_input, (224, 224))
        densenet_label, densenet_confidence = interpret_predictions(densenet_predictions)
        st.write(f"Prediction: {densenet_label} (Confidence: {densenet_confidence:.2f})")
