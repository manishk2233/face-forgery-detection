import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np

# Load the model
@st.cache(allow_output_mutation=True)
def load_keras_model(model_path):
    return load_model(model_path)

# Display model summary
def display_model_summary(model):
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    summary_str = "\n".join(summary_list)
    st.text(summary_str)

# Main function for the Streamlit app
def main():
    st.title("Keras Model Checker")

    # Model upload
    model_path = st.text_input("densenet121.keras", "model.keras")
    
    if st.button("Load Model"):
        try:
            model = load_keras_model(model_path)
            st.success("Model loaded successfully!")
            
            # Display the model summary
            st.subheader("Model Summary")
            display_model_summary(model)
            
            # Display the input image size
            input_shape = model.input_shape[1:4]  # Skip the batch size dimension
            st.subheader("Model Input Image Size")
            st.text(f"Input shape: {input_shape}")
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    # Image upload and preprocessing
    st.subheader("Upload and Test Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        img_array = np.array(image.resize(input_shape[:2]))  # Resize to match the input shape of the model
        img_array = img_array / 255.0  # Rescale the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict with the model
        prediction = model.predict(img_array)
        st.subheader("Model Prediction")
        st.text(f"Prediction: {prediction[0][0]}")

if __name__ == "__main__":
    main()
