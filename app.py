import tensorflow as tf
model = tf.keras.models.load_model('resnet50.h5')
model = tf.keras.models.load_model('xception.h5')
model.save('resnet50.keras')
model.save('xception.keras')
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

# Load the trained models
densenet_model_path = 'densenet121.keras'
xception_model_path = 'xception.keras'
resnet50_model_path = 'resnet50.keras'

try:
    densenet_model = tf.keras.models.load_model(densenet_model_path)
    st.success("DenseNet121 model loaded successfully.")
except Exception as e:
    st.error(f"Error loading DenseNet121 model: {e}")

try:
    xception_model = tf.keras.models.load_model(xception_model_path)
    st.success("Xception model loaded successfully.")
except Exception as e:
    st.error(f"Error loading Xception model: {e}")

try:
    resnet50_model = tf.keras.models.load_model(resnet50_model_path)
    st.success("ResNet50 model loaded successfully.")
except Exception as e:
    st.error(f"Error loading ResNet50 model: {e}")

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
st.title("Deepfake Detection with DenseNet121, Xception, and ResNet50")

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

    # Predict using DenseNet121
    if densenet_model:
        st.subheader("DenseNet121 Prediction")
        densenet_predictions = preprocess_and_predict(temp_file_path, densenet_model, densenet_preprocess_input, (128, 128))
        densenet_label, densenet_confidence = interpret_predictions(densenet_predictions)
        st.write(f"Prediction: {densenet_label} (Confidence: {densenet_confidence:.2f})")

    # Predict using Xception
    if xception_model:
        st.subheader("Xception Prediction")
        xception_predictions = preprocess_and_predict(temp_file_path, xception_model, xception_preprocess_input, (128, 128))
        xception_label, xception_confidence = interpret_predictions(xception_predictions)
        st.write(f"Prediction: {xception_label} (Confidence: {xception_confidence:.2f})")

    # Predict using ResNet50
    if resnet50_model:
        st.subheader("ResNet50 Prediction")
        resnet50_predictions = preprocess_and_predict(temp_file_path, resnet50_model, resnet_preprocess_input, (128, 128))
        resnet50_label, resnet50_confidence = interpret_predictions(resnet50_predictions)
        st.write(f"Prediction: {resnet50_label} (Confidence: {resnet50_confidence:.2f})")

    # Final decision based on majority voting
    labels = [densenet_label, xception_label, resnet50_label]
    final_label = max(set(labels), key=labels.count)
    st.subheader(f"Final Prediction: {final_label}")
