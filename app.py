import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Paths to saved model files
model_path_xception = 'path_to_xception_model.h5'  # Use complete model path if needed
model_path_resnet = 'path_to_resnet_model.h5'      # Use complete model path if needed

# Define class labels
class_labels = ['Fake', 'Real']  # Adjust these labels according to your dataset

# Function to define the Xception model
def get_xception_model(input_shape=(128, 128, 3), num_classes=2):
    # Input layer
    input = tf.keras.Input(shape=input_shape)
    
    # Load Xception with ImageNet weights without the top dense layers
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
    
    # Load ResNet50 with ImageNet weights without the top dense layers
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

# Load the models with weights
try:
    # Option 1: Load entire model (if model was saved entirely)
    # model_xception = load_model(model_path_xception)
    # model_resnet = load_model(model_path_resnet)

    # Option 2: Load weights into defined architecture
    model_xception = get_xception_model()
    model_xception.load_weights(model_path_xception)

    model_resnet = get_resnet50_model()
    model_resnet.load_weights(model_path_resnet)
    
    st.success("Models loaded successfully.")
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")

# Function to preprocess the image
def preprocess_image(img, model_name):
    img = img.resize((128, 128))  # Resize image to 128x128 pixels
    img_array = keras_image.img_to_array(img)  # Correct use of keras_image
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
st.write("Upload an image to classify it as Real or Fake using both Xception and ResNet50 models.")

# File uploader for single image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make a prediction with both models
    with st.spinner('Classifying...'):
        label_xception, confidence_xception = predict_image(image, model_xception, 'Xception')
        label_resnet, confidence_resnet = predict_image(image, model_resnet, 'ResNet50')

        # Display results
        st.write("### Xception Model")
        st.write(f"Prediction: **{label_xception}**")
        st.write(f"Confidence: **{confidence_xception * 100:.2f}%**")
        
        st.write("### ResNet50 Model")
        st.write(f"Prediction: **{label_resnet}**")
        st.write(f"Confidence: **{confidence_resnet * 100:.2f}%**")
