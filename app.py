import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

# Function to load and adjust models
def load_and_adjust_model(model_path):
    model = tf.keras.models.load_model(model_path)
    
    # Display the original model architecture for debugging purposes
    model.summary()
    
    # Assuming that the last Dense layer is causing issues, we pop it off and add a new one
    # Pop layers until the last layer is compatible with new input/output structure
    model.layers.pop()  # Remove the last layer (which is likely problematic)
    model.layers.pop()  # Remove additional layers if necessary

    # Adding new layers
    x = model.layers[-1].output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Assuming binary classification

    # Create a new model
    new_model = tf.keras.models.Model(inputs=model.input, outputs=x)

    # Save the adjusted model (optional)
    adjusted_model_path = model_path.replace('.h5', '_adjusted.h5')
    new_model.save(adjusted_model_path)
    return new_model

# Load and adjust models
resnet50_model = load_and_adjust_model('resnet50.h5')
xception_model = load_and_adjust_model('xception.h5')
densenet_model = load_and_adjust_model('densenet121.h5')

# Streamlit UI
st.title("Deepfake Detection App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((128, 128))  # Resize to match the input size of the models
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using each model
    resnet50_pred = resnet50_model.predict(img_array)[0][0]
    xception_pred = xception_model.predict(img_array)[0][0]
    densenet_pred = densenet_model.predict(img_array)[0][0]

    # Combine predictions (simple average for this example)
    final_pred = (resnet50_pred + xception_pred + densenet_pred) / 3

    # Display predictions
    st.write(f"ResNet50 Prediction: {resnet50_pred:.4f}")
    st.write(f"Xception Prediction: {xception_pred:.4f}")
    st.write(f"DenseNet121 Prediction: {densenet_pred:.4f}")
    st.write(f"Final Prediction: {final_pred:.4f}")

    if final_pred > 0.5:
        st.write("The image is predicted as **Fake**")
    else:
        st.write("The image is predicted as **Real**")
