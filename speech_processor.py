import speech_recognition
import pyaudio
from pydub import AudioSegment
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recognize_from_microphone():
    """Capture audio from microphone with basic preprocessing"""
    try:
        # Initialize recognizer and microphone
        recognizer = speech_recognition.Recognizer()
        microphone = speech_recognition.Microphone()
        
        logger.info("Initialized speech recognition")
        
        # Use microphone
        with microphone as source:
            logger.info("Adjusting for ambient noise... Please be quiet.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            logger.info("Ready! Speak now...")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                
                # Save audio with standard settings
                temp_path = "temp_recording.wav"
                with open(temp_path, "wb") as f:
                    f.write(audio.get_wav_data())
                
                # Enhance the recorded audio
                audio_segment = AudioSegment.from_wav(temp_path)
                enhanced_audio = enhance_audio_for_waray(audio_segment)
                enhanced_path = "temp_enhanced.wav"
                enhanced_audio.export(enhanced_path, format="wav")
                
                # Clean up original temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                return enhanced_path
                
            except speech_recognition.WaitTimeoutError:
                logger.warning("No speech detected. Please try again.")
                return None
                
    except Exception as e:
        logger.error(f"Error recording audio: {e}")
        if os.path.exists("temp_recording.wav"):
            os.remove("temp_recording.wav")
        if os.path.exists("temp_enhanced.wav"):
            os.remove("temp_enhanced.wav")
        return None

def enhance_audio_for_waray(audio_segment):
    """Basic audio enhancement for Waray speech"""
    try:
        audio = audio_segment.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio = audio.normalize()
        return audio
    except Exception as e:
        logger.error(f"Error enhancing audio: {e}")
        return audio_segment