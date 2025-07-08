import streamlit as st
import librosa 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import base64

# Function to encode background image
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background image
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-position: center;
    background-size: cover;
    }}
    </style>
    '''
    st.markdown('<style>h1 { color: white; }</style>', unsafe_allow_html=True)
    st.markdown('<style>p { color: white; font-weight: bold; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('C:/Users/palak/Downloads/img.jpg')

def predict_genre(model, audio_file_path, genre_mapping):
    """Predicts the genre of an audio file using a trained model.

    Args:
        model: Trained genre classification model.
        audio_file_path: Path to the audio file.
        genre_mapping: Dictionary mapping class indices to genre labels.

    Returns:
        Predicted genre label.
    """

    # Load audio file
    signal, sample_rate = librosa.load(audio_file_path, sr=SAMPLE_RATE)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T

    # Reshape MFCCs to match model input shape
    mfcc = mfcc[:130, ...]  # Take only the first 130 MFCCs
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add extra dimensions

    # Predict using the model
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)
    

    # Map predicted index to genre label
    genre_label = genre_mapping[predicted_index[0]]
    

    return genre_label

# Load your trained model
model_path = "C:/Users/palak/Downloads/DetectionModel.h5"  # Update the path to your model file
model = load_model(model_path)

# Genre mapping (update this according to your dataset)
genre_mapping = {0: "Real", 1: "Fake"}

# Set the sample rate used for loading the audio files
SAMPLE_RATE = 22050  # Change according to your model requirements

# Streamlit interface
st.title("DeepFake Voice Detection")
st.write("Upload an audio file to check if it's real or fake")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3","flac"])

if audio_file is not None:
    with open("temp_audio_file", "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Make the prediction
    predicted_genre = predict_genre(model, "temp_audio_file", genre_mapping)
    
    st.write("Predicted genre:", predicted_genre)