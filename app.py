import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Load the trained models
model_path_xception = 'xception.h5'  # Replace with the actual path
model_path_resnet = 'resnet50.h5'      # Replace with the actual path

# Function to define the Xception model
def get_xception_model(input_shape=(128, 128, 3), num_classes=2):
    # Input layer
    input = tf.keras.Input(shape=input_shape)
    
    # Load Xception with imagenet weights without the top dense layers
    xception_base = Xception(weights='imagenet', include_top=False, input_tensor=input)
    
    # Adding layers on top of Xception
    x = tf.keras.layers.GlobalAveragePooling2D()(xception_base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Creating the model
    model = tf.keras.Model(inputs=xception_base.input, outputs=output)
    return model

# Function to define the ResNet50 model
def get_resnet50_model(input_shape=(128, 128, 3), num_classes=2):
    # Input layer
    input = tf.keras.Input(shape=input_shape)
    
    # Load ResNet50 with imagenet weights without the top dense layers
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_tensor=input)
    
    # Adding layers on top of ResNet50
    x = tf.keras.layers.GlobalAveragePooling2D()(resnet_base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Creating the model
    model = tf.keras.Model(inputs=resnet_base.input, outputs=output)
    return model

# Load models
model_xception = get_xception_model()
model_xception.load_weights(model_path_xception)

model_resnet = get_resnet50_model()
model_resnet.load_weights(model_path_resnet)

# Define class labels
class_labels = ['Fake', 'Real']  # Adjust these labels according to your dataset

# Function to preprocess the image
def preprocess_image(img, model_name):
    img = img.resize((128, 128))  # Resize image to 128x128 pixels
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Preprocess image according to the selected model
    if model_name == 'Xception':
        return preprocess_input_xception(img_array)
    elif model_name == 'ResNet50':
        return preprocess_input_resnet(img_array)

# Function to make predictions
def predict_image(img, model, model_name):
    img_array = preprocess_image(img, model_name)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class]
    return predicted_label, confidence

# Streamlit UI
st.title("Real vs Fake Image Detection")
st.write("Upload an image to classify it as Real or Fake.")

# Model selection
model_option = st.selectbox(
    "Select the model to use:",
    ("Xception", "ResNet50")
)

# Load the selected model
selected_model = model_xception if model_option == "Xception" else model_resnet

# Confidence threshold
confidence_threshold = st.slider(
    "Confidence threshold:", 0.0, 1.0, 0.5, 0.01
)

# File uploader for single image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Batch file uploader for multiple images
uploaded_files = st.file_uploader("Choose multiple images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

def display_result(label, confidence):
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
    if confidence < confidence_threshold:
        st.write(f"⚠️ Confidence is below the threshold of {confidence_threshold * 100:.2f}%.")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make a prediction
    with st.spinner('Classifying...'):
        label, confidence = predict_image(image, selected_model, model_option)
        display_result(label, confidence)

if uploaded_files:
    st.write("Batch Prediction Results:")
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=f'Image: {file.name}', use_column_width=True)
        
        # Make a prediction for each image
        with st.spinner(f'Classifying {file.name}...'):
            label, confidence = predict_image(image, selected_model, model_option)
            display_result(label, confidence)
