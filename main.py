import os
import numpy as np
import tensorflow as tf
from audio_processing import is_valid_audio, extract_features
from speech_processor import recognize_from_microphone, enhance_audio_for_waray

class SpeechProcessor:
    def __init__(self):
        try:
            print("Loading model and encoder...")
            self.model = tf.keras.models.load_model('waray_speech_model.keras')
            self.encoder_classes = np.load('encoder_classes.npy', allow_pickle=True)
            print("Model and encoder loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict_speech(self, file_path):
        try:
            features = extract_features(file_path)
            if features is None:
                return None, 0.0
            features = np.expand_dims(features, axis=0)
            prediction = self.model.predict(features, verbose=0)
            confidence = np.max(prediction)
            label = self.encoder_classes[np.argmax(prediction)]
            return label, confidence
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, 0.0

def process_speech():
    try:
        speech_processor = SpeechProcessor()
        print("\nListening...")
        audio_file = recognize_from_microphone()
        
        if not audio_file:
            print("No audio recorded. Try again.")
            return
            
        if not is_valid_audio(audio_file):
            print("Invalid or unclear recording. Try again.")
            return
            
        label, confidence = speech_processor.predict_speech(audio_file)
        if label:
            print(f"\nRecognized Word: {label} (Confidence: {confidence:.2f})")
        else:
            print("Could not recognize speech. Please try again.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)

if __name__ == '__main__':
    print("Starting speech recognition system...")
    print("Supported words: aga, maupay, gab-i, ngaran")
    
    while True:
        word = input("\nEnter a word to test or 'quit' to exit: ")
        if word.lower() == 'quit':
            break
        process_speech()
