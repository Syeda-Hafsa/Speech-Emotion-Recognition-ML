import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pyaudio
import wave
import matplotlib.pyplot as plt

 #- Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        return result

# - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

# - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

# - Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("RAVDESS\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
# - Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

#- Get the shape of the training and testing datasets

print((x_train.shape[0], x_test.shape[0]))

# - Get the number of features extracted

print(f'Features extracted: {x_train.shape[1]}')

# - Initialize the Multi Layer Perceptron Classifier

model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# - Train the model
model.fit(x_train,y_train)

#DataFlair - Predict for the test set
y_pred=model.predict(x_test)

#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))




# ... (previous code)

# Function to record audio from the microphone
def record_audio(filename, duration=5):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=44100, input=True,
                        frames_per_buffer=1024)

    print("Recording...")
    frames = []

    for _ in range(0, int(44100 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording!")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))

# Function to predict emotion from a recorded audio file
def predict_emotion_from_audio(file_name):
    feature = extract_feature(file_name, mfcc=True, chroma=True, mel=True)
    emotion = model.predict([feature])[0]
    return emotion

# ... (previous code)

# Record audio from the microphone
audio_file_name = "input_audio.wav"
record_audio(audio_file_name)

# Predict emotion from the recorded audio
predicted_emotion = predict_emotion_from_audio(audio_file_name)

# Print the predicted emotion
print("Predicted Emotion:", predicted_emotion)

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

# ... (previous code)

# Predict emotion from the recorded audio
predicted_emotion = predict_emotion_from_audio(audio_file_name)

# Print the predicted emotion as an emoji
if predicted_emotion in emotion_to_emoji:
    emoji = emotion_to_emoji[predicted_emotion]
    print("Predicted Emotion:", predicted_emotion, emoji)
else:
    print("Predicted Emotion:", predicted_emotion)





# ... (your existing code)

# Function to visualize emotion output
def visualize_emotion_output(y_pred, labels):
    # Count the occurrences of each emotion in the predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    emotion_counts = dict(zip(unique, counts))

    # Create an array of counts in the order of observed_emotions
    emotion_counts_ordered = [emotion_counts[emotion] for emotion in labels]

    # Plot the emotion distribution as a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(labels, emotion_counts_ordered)
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title('Emotion Distribution')
    plt.show()

# ... (your existing code)

# Visualize the emotion output
visualize_emotion_output(y_pred, labels=observed_emotions)

import tkinter as tk
from tkinter import Label, PhotoImage
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pyaudio
import wave
import matplotlib.pyplot as plt

# Rest of your code...

# Function to display emoji in a GUI window
def display_emoji(emotion):
    root = tk.Tk()
    root.title("Emotion Emoji")

    # Define emoji images for each emotion
    emoji_images = {
        'calm': '😌',
        'happy': '😄',
        'fearful': '😨',
        'disgust': '😖',
        'angry': '😡',
        'sad': '😢',
        'neutral': '😐',
        'surprised': '😮'
    }

    emoji_label = Label(root, text=emoji_images.get(emotion, '😐'), font=("Arial", 72))
    emoji_label.pack(pady=20)

    root.mainloop()

# ... (previous code)

# Predict emotion from the recorded audio
predicted_emotion = predict_emotion_from_audio(audio_file_name)

# Print the predicted emotion
#print("Predicted Emotion:", predicted_emotion)

# Display emoji in a GUI window
display_emoji(predicted_emotion)

# ... (your existing code)
