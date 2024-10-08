import streamlit as st
import requests
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set the page configuration with a custom icon
st.set_page_config(
    page_title="Cotton Disease Detection",
    page_icon="🌿",  # Custom leaf icon
)

# Define the URL for the model file on GitHub
MODEL_URL = 'https://github.com/HamzaMLDS/Cotton-Disease-Detection/blob/4239ee7e5bd4165e9d607c7a5309bb628cebff60/cotton_disease_model.h5?raw=true'

# Define the local path to save the downloaded model
MODEL_PATH = 'cotton_disease_model.h5'

# Function to download the model file from GitHub
def download_model(url, local_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for errors
    with open(local_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

# Download the model if it does not exist or is smaller than expected
if not os.path.isfile(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100:
    download_model(MODEL_URL, MODEL_PATH)

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
    class_names = ['Bacterial Blight🌿😢', 'Curl Virus🌿😢', 'Fusarium Wilt🌿😢', 'Healthy🌿😄']
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]

# Inject custom CSS to remove the background and style emojis
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    header {
        background-color: black !important;
    }
    .css-1q1n0ol {
        background-color: black !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00cc44;
    }
    .stButton button {
        background-color: #00cc44;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: black;
        color: #00cc44;
    }
    /* Style the profile links and emojis */
    .profile-links {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        margin-top: 20px;
    }
    .profile-links a {
        font-size: 30px;
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App
st.title('🌿 Welcome to Cotton Disease Detection 🌿')
st.write("""
This app helps in identifying common diseases in cotton leaves. 
Simply upload an image of a cotton leaf, and the model will predict whether the leaf is affected by one of the following diseases:
- **Bacterial Blight**
- **Curl Virus**
- **Fusarium Virus**
- Or if the leaf is **Healthy**
""")

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

# Personal introduction and contact information
st.markdown("---")
st.write("## About Me")
st.write("""
Hello! I'm **Hamza Rehman**, a student of **BS Computer Science (BSCS)**. I'm currently learning machine learning and working on exciting projects like this one. My goal is to gain deep insights into AI and apply them to solve real-world problems.
""")
st.write("### Connect with Me")

# Display clickable contact information with larger emojis
st.markdown(
    """
    <div class="profile-links">
    <a href="mailto:mhrwork2001@gmail.com" target="_blank">📧 Email</a>  
    <a href="https://www.linkedin.com/in/hamza-rehman-4aa1a323a/" target="_blank">🔗 LinkedIn</a>  
    <a href="https://github.com/HamzaMLDS" target="_blank">💻 GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
