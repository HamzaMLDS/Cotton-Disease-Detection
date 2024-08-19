import streamlit as st
import requests
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Define the URL for the model file on Dropbox
MODEL_URL = 'https://www.dropbox.com/scl/fi/02pwzjlm024cvig1wezzj/cotton_disease_model.keras?rlkey=rzcstdhtjujk24m3vhvxyzr74&st=ocuzyxf6&dl=1'

# Define the local path to save the downloaded model
MODEL_PATH = 'cotton_disease_model.h5'

# Function to download the model file from Dropbox
def download_model(url, local_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for errors
    with open(local_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

# Download the model if it does not exist
if not os.path.isfile(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100:
    st.write('Downloading model...')
    download_model(MODEL_URL, MODEL_PATH)
    st.write('Model downloaded.')

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define a function to interpret the model's prediction
def interpret_prediction(prediction):
    # Mapping the predicted class to disease names
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
    prediction = model.predict(processed_image)
    
    # Interpret and display the prediction
    result = interpret_prediction(prediction)
    st.write('Prediction:', result)
