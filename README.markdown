# Deepfake Voice Detection

This project implements a deep learning model to detect whether an audio file contains a real or fake (deepfake) voice using Mel-frequency cepstral coefficients (MFCCs) and a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. The model is trained to classify audio files as either "Real" or "Fake" and is deployed through a user-friendly Streamlit web interface.

## Project Overview
- **Objective**: Classify audio files as real or fake using MFCC features and an LSTM-based neural network.
- **Model**: RNN with LSTM layers trained on a dataset of audio MFCC features.
- **Interface**: A Streamlit web application for uploading audio files and viewing predictions.
- **Dataset**: Expects a JSON file (`data.json`) containing MFCC features and labels (not included in this repository).

## Requirements
To run this project, install the required Python packages listed below:

Required libraries:
- `streamlit`
- `librosa`
- `numpy`
- `tensorflow`



## Usage
1. **Run the Streamlit app**:
   ```bash
   streamlit run temp.py
   ```

2. **Upload an audio file**:
   - Open the Streamlit web interface in your browser (typically at `http://localhost:8501`).
   - Upload an audio file (supported formats: `.wav`, `.mp3`, `.flac`).
   - The app will process the audio and display whether the voice is classified as "Real" or "Fake."

## Project Structure
- `temp.py`: Main Streamlit application script for the web interface and prediction logic.
- `DeepfakeVoiceDetectionModel.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `DetectionModel.h5`: Pre-trained model file.
- `img.jpg`: Background image for the Streamlit app.
- `data.json`: Expected dataset file with MFCC features and labels.

## Model Details
- **Architecture**: The model is an RNN with two LSTM layers (64 units each), followed by a dense layer (64 units, ReLU activation), a dropout layer (0.3), and an output layer (2 units, softmax activation).
- **Input**: MFCC features extracted from audio files with a sample rate of 22,050 Hz, 13 MFCC coefficients, and a fixed length of 130 frames.
- **Training**: Trained for 40 epochs with Adam optimizer (learning rate 0.0001) and sparse categorical crossentropy loss.
- **Output**: Binary classification ("Real" or "Fake").

## How It Works
1. **Feature Extraction**: Audio files are processed using `librosa` to extract MFCC features, which are then reshaped to match the model’s input requirements.
2. **Prediction**: The trained LSTM model predicts the class (Real or Fake) based on the MFCC features.
3. **Web Interface**: The Streamlit app allows users to upload audio files, processes them, and displays the prediction.

