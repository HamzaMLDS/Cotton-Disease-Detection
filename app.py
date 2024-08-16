
import streamlit as st
from PIL import Image
import numpy as np
import requests
import os
from tensorflow.keras.models import load_model 
# Function to download the model file from a URL
def download_model(url, output_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        with open(output_path, 'wb') as f:
            f.write(response.content)
        st.success("Model downloaded successfully.")
    except Exception as e:
        st.error(f"Error downloading the model: {e}")

# Define the model path and download if necessary
model_url = 'https://drive.google.com/uc?export=download&id=15kKlG9rJSj8oxM3dFeCrEIjRagjyf-MI'
MODEL_PATH = 'cotton_disease_model.h5'

if not os.path.exists(MODEL_PATH):
    st.write("Downloading the model...")
    download_model(model_url, MODEL_PATH)

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define a function to interpret the model's prediction
def interpret_prediction(prediction):
    class_names = ['Bacterial Blight', 'Curl Virus', 'Fusarium Virus', 'Healthy']
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]

# Define the Streamlit app
st.title('Cotton Disease Detection')
st.write('Upload an image of a cotton leaf to detect the disease.')

uploaded_file = st.file_uploader('Choose an image...', type='jpg')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write('Classifying...')
    processed_image = preprocess_image(image)
    try:
        prediction = model.predict(processed_image)
        # Interpret and display the prediction
        result = interpret_prediction(prediction)
        st.write('Prediction:', result)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
