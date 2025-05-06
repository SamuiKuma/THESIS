import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import random
from pydub import AudioSegment

Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

# Function to extract MFCC (audio enhancement/manipulation features)
def extract_features(file_path, max_pad_len=100):
    try:
        audio, sample_rate = sf.read(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to augment audio
def augment_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        # Apply random pitch shift
        pitch_shift = random.uniform(-2.0, 2.0)
        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * (2.0 ** (pitch_shift / 12.0)))})
        audio = audio.set_frame_rate(audio.frame_rate)
        # Apply random speed change
        speed_change = random.uniform(0.9, 1.1)
        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed_change)})
        audio = audio.set_frame_rate(audio.frame_rate)
        # Apply random noise
        noise = AudioSegment.silent(duration=len(audio))
        noise = noise + random.uniform(-30, -20)
        audio = audio.overlay(noise)
        # Export augmented audio to a temporary file
        augmented_path = "augmented_temp.wav"
        audio.export(augmented_path, format="wav")
        return augmented_path
    except Exception as e:
        print(f"Error augmenting {file_path}: {e}")
        return None

# Load dataset metadata
def load_dataset(csv_path, audio_dir):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None, None
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV file with {len(df)} entries")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None

    audio_data = []
    labels = []
    for _, row in df.iterrows():
        file_path = os.path.join(audio_dir, row['filename'])
        print(f"Processing file: {file_path}")
        label = row['text']
        # Extract features from original audio
        features = extract_features(file_path)
        if features is not None:
            audio_data.append(features)
            labels.append(label)
        else:
            print(f"Failed to process file: {file_path}")
        # Extract features from augmented audio
        augmented_path = augment_audio(file_path)
        if augmented_path is not None:
            augmented_features = extract_features(augmented_path)
            if augmented_features is not None:
                audio_data.append(augmented_features)
                labels.append(label)
            else:
                print(f"Failed to process augmented file: {augmented_path}")
    return np.array(audio_data), np.array(labels)

# Define dataset paths using raw string (r"") to avoid Unicode errors
dataset_path = r"D:\lstm\labels.csv"
audio_folder = r"D:\lstm\audio_wav"

'''
# Print paths for debugging
print(f"Dataset path: {dataset_path}")
print(f"Audio folder: {audio_folder}")

for debug purposes only
'''

# Load and preprocess data
X, y = load_dataset(dataset_path, audio_folder)

if X is None or y is None:
    print("Failed to load dataset. Exiting...")
    exit()

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = tf.keras.utils.to_categorical(y, num_classes=len(encoder.classes_))

# Reshape data for LSTM
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
def build_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the model with early stopping
model = build_model((X_train.shape[1], X_train.shape[2]), len(encoder.classes_))

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save model
model.save("waray_speech_model.keras")

# Predict function
def predict_speech(file_path, model, encoder):
    features = extract_features(file_path)
    if features is None:
        return "Error processing audio"
    
    # Print feature shape for debugging
    print(f"Extracted features shape: {features.shape}")
    
    # Ensure the features have the correct shape
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    print(f"Features shape after expanding dimensions: {features.shape}")
    
    # Make prediction
    prediction = model.predict(features)
    print(f"Prediction raw output: {prediction}")
    
    # Decode the predicted label
    predicted_label = encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Real-time microphone input
def recognize_from_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak something...")
        audio = recognizer.listen(source)
        with open("temp.wav", "wb") as f:
            f.write(audio.get_wav_data())
        return "temp.wav"

# Example usage with real-time speech input
audio_file = recognize_from_microphone()
print("Predicted:", predict_speech(audio_file, model, encoder))


