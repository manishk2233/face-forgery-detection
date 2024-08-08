import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import DenseNet121

# Load the pre-trained Keras model
model = tf.keras.models.load_model('model.keras')  # Make sure this is the path to your .keras model

# Set title of the app
st.title("Real vs Fake Image Classifier")

# Instructions for the app
st.write("""
         Upload an image to determine whether it is **Real** or **Fake**.
         """)

# Function to load and preprocess the image
def load_and_preprocess_image(image, target_size=(128, 128)):
    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    return img_array

# Image upload functionality
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img_array = load_and_preprocess_image(image)
    
    # Make predictions
    predictions = model.predict(img_array)
    confidence_score = predictions[0][0]
    
    # Display the results
    if confidence_score > 0.5:
        st.write(f"Prediction: **Real** (Confidence: {confidence_score:.2f})")
    else:
        st.write(f"Prediction: **Fake** (Confidence: {1 - confidence_score:.2f})")
