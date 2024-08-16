import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('cotton_disease_model.h5')

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
