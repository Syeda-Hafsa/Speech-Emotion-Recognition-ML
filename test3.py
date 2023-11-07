import streamlit as st
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
import pyaudio
import wave
import matplotlib.pyplot as plt

# Function to extract audio features
def extract_feature(file_name, mfcc, chroma, mel):
    # Your existing feature extraction code here

# Function to predict emotion from audio
  def predict_emotion_from_audio(file_name):
    feature = extract_feature(file_name, mfcc=True, chroma=True, mel=True)
    emotion = model.predict([feature])[0]
    return emotion

# Function to record audio
def record_audio(filename, duration=5):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=44100, input=True,
                        frames_per_buffer=1024)

    st.text("Recording...")

    frames = []

    for _ in range(0, int(44100 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    st.text("Finished recording!")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))

# Initialize MLPClassifier and load the trained model (you should load your trained model here)
model = MLPClassifier()

# Streamlit app
st.title("Emotion Recognition App")

# Record audio button
if st.button("Record Audio"):
    audio_file_name = "input_audio.wav"
    record_audio(audio_file_name)
    predicted_emotion = predict_emotion_from_audio(audio_file_name)

    # Define a dictionary to map emotions to emojis
    emotion_to_emoji = {
        'calm': '😌',
        'happy': '😄',
        'fearful': '😨',
        'disgust': '😖',
        'angry': '😡',
        'sad': '😢',
        'neutral': '😐',
        'surprised': '😮'
    }

    # Predict emotion from the recorded audio
    if predicted_emotion in emotion_to_emoji:
        emoji = emotion_to_emoji[predicted_emotion]
        st.write("Predicted Emotion:", predicted_emotion, emoji)
    else:
        st.write("Predicted Emotion:", predicted_emotion)
